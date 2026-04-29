"""
SysVec Trainer
==============
Implements the DPO-style optimisation objective from Section 3.4 of the paper.

Preference data construction (paper eq. 4):
    yw = f_theta(s ⊕ x)   — response WITH textual system prompt
    yl = f_theta(x)        — response WITHOUT any system prompt

Optimisation objective (paper eq. 5):

    min_{vsys}  -E_{(x, yw, yl) ~ D} [
        log σ( β * log[ p_vsys(yw) / p_base(yw) ]
                - β * log[ p_vsys(yl) / p_base(yl) ] )
    ]

Where:
    p_vsys(y)  = probability of sequence y when vsys is injected
    p_base(y)  = probability of sequence y WITHOUT injection (reference)
    β          = DPO temperature (controls how strongly we push toward yw)
"""

import os
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional
from tqdm import tqdm

from .model import SysVec


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """
    Holds tokenised (yw, yl) preference pairs.

    Each item:
        yw_input_ids      — token ids of the FULL sequence (system + user + response)
        yw_attention_mask
        yl_input_ids      — token ids of (user + response) only
        yl_attention_mask
    """

    def __init__(self, samples: List[Dict]):
        """
        Args:
            samples: list of dicts with keys
                     'yw_input_ids', 'yw_attention_mask',
                     'yl_input_ids', 'yl_attention_mask'
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad variable-length sequences within a batch."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        tensors = [item[k] for item in batch]
        # Determine pad value: 0 for masks, pad_token_id for ids
        pad_val = 0 if "mask" in k else batch[0].get("pad_token_id", 0)
        max_len = max(t.size(0) for t in tensors)
        padded = torch.stack(
            [F.pad(t, (0, max_len - t.size(0)), value=pad_val) for t in tensors]
        )
        out[k] = padded
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data synthesis helper  (offline; requires a GPU-accessible base model)
# ─────────────────────────────────────────────────────────────────────────────

def build_preference_samples(
    sysvec_model: SysVec,
    system_prompt: str,
    user_questions: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> List[Dict]:
    """
    Generate (yw, yl) pairs using the base model.

        yw = tokenise( system_prompt + user_question + model_response_with_sp )
        yl = tokenise( user_question + model_response_without_sp )

    Both yw and yl are encoded as full sequences (prompt + completion) because
    the DPO loss needs log p over the *entire* sequence.

    Returns list of dicts ready for PreferenceDataset.
    """
    tokenizer = sysvec_model.tokenizer
    model = sysvec_model
    device = sysvec_model.device
    samples = []

    for i in range(0, len(user_questions), batch_size):
        batch_qs = user_questions[i: i + batch_size]

        # ── Generate yw: response WITH system prompt ─────────────────────
        yw_inputs = tokenizer(
            [_format_with_system(tokenizer, system_prompt, q) for q in batch_qs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.inference_mode():
            yw_out = model.generate(
                yw_inputs.input_ids,
                attention_mask=yw_inputs.attention_mask,
                inject=False,          # base model only during data gen
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # ── Generate yl: response WITHOUT system prompt ──────────────────
        yl_inputs = tokenizer(
            [_format_without_system(tokenizer, q) for q in batch_qs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.inference_mode():
            yl_out = model.generate(
                yl_inputs.input_ids,
                attention_mask=yl_inputs.attention_mask,
                inject=False,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # ── Package into samples ─────────────────────────────────────────
        for j in range(len(batch_qs)):
            yw_ids = yw_out[j].cpu()
            yl_ids = yl_out[j].cpu()

            yw_mask = (yw_ids != tokenizer.pad_token_id).long()
            yl_mask = (yl_ids != tokenizer.pad_token_id).long()

            samples.append({
                "yw_input_ids": yw_ids,
                "yw_attention_mask": yw_mask,
                "yl_input_ids": yl_ids,
                "yl_attention_mask": yl_mask,
                "pad_token_id": tokenizer.pad_token_id,
            })

        print(f"  Generated {min(i + batch_size, len(user_questions))}/{len(user_questions)} samples")

    return samples


def _format_with_system(tokenizer, system_prompt: str, user_query: str) -> str:
    """Apply chat template with a system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without a chat template
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_query}\n<|assistant|>\n"


def _format_without_system(tokenizer, user_query: str) -> str:
    """Apply chat template without any system prompt."""
    messages = [{"role": "user", "content": user_query}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<|user|>\n{user_query}\n<|assistant|>\n"


# ─────────────────────────────────────────────────────────────────────────────
# DPO Loss
# ─────────────────────────────────────────────────────────────────────────────

def dpo_loss(
    log_prob_vsys_yw: torch.Tensor,   # [B]  log p_vsys(yw)
    log_prob_base_yw: torch.Tensor,   # [B]  log p_base(yw)
    log_prob_vsys_yl: torch.Tensor,   # [B]  log p_vsys(yl)
    log_prob_base_yl: torch.Tensor,   # [B]  log p_base(yl)
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Paper eq. 5:

        L = -E[ log σ( β*(log p_vsys(yw) - log p_base(yw))
                       - β*(log p_vsys(yl) - log p_base(yl)) ) ]
    """
    pi_yw = log_prob_vsys_yw - log_prob_base_yw   # log-ratio for preferred
    pi_yl = log_prob_vsys_yl - log_prob_base_yl   # log-ratio for dispreferred

    logits = beta * (pi_yw - pi_yl)
    loss = -F.logsigmoid(logits).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SysVecTrainer:
    """
    Trains the SysVec vector using the DPO-style objective.

    Paper hyperparameters (Section 4.1):
        lr          = 5e-4
        weight_decay= 0.05
        optimizer   = AdamW
        scheduler   = CosineAnnealingLR with 100 warmup steps
        batch_size  = 8 (with gradient accumulation)
        epochs      = 25 (Llama-2/3),  5 (Mistral)
        beta        = 0.1  (standard DPO)
    """

    def __init__(
        self,
        sysvec_model: SysVec,
        dataset: PreferenceDataset,
        output_dir: str = "./checkpoints",
        lr: float = 5e-4,
        weight_decay: float = 0.05,
        beta: float = 0.1,
        epochs: int = 25,
        batch_size: int = 2,
        grad_accum_steps: int = 4,
        warmup_steps: int = 100,
    ):
        self.model = sysvec_model
        self.dataset = dataset
        self.output_dir = output_dir
        self.beta = beta
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.effective_batch = batch_size * grad_accum_steps

        os.makedirs(output_dir, exist_ok=True)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Only optimise vsys — base model is frozen
        self.optimizer = AdamW(
            [self.model.vsys],
            lr=lr,
            weight_decay=weight_decay,
        )

        total_steps = epochs * math.ceil(len(self.loader) / grad_accum_steps)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Warmup handled manually (linear ramp)
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def _warmup_lr(self):
        if self.global_step < self.warmup_steps:
            factor = (self.global_step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * factor if "initial_lr" in pg else pg["lr"] * factor

    def train(self):
        """Run the full training loop."""
        self.model.base_model.eval()        # base weights always frozen
        self.model.vsys.requires_grad_(True)

        # Store initial LR for warmup
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        print(f"\n[SysVecTrainer] Starting training for {self.epochs} epochs")
        print(f"  Dataset size     : {len(self.dataset)}")
        print(f"  Effective batch  : {self.effective_batch}")
        print(f"  Injection layer  : {self.model.injection_layer}")
        print(f"  Alpha            : {self.model.alpha}")
        print(f"  Beta (DPO)       : {self.beta}\n")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad()

            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for step, batch in enumerate(pbar):
                device = self.model.device
                yw_ids  = batch["yw_input_ids"].to(device)
                yw_mask = batch["yw_attention_mask"].to(device)
                yl_ids  = batch["yl_input_ids"].to(device)
                yl_mask = batch["yl_attention_mask"].to(device)

                # ── log p_vsys(yw) and log p_vsys(yl) (with grad) ────────
                self.model.train()
                lp_vsys_yw = self.model.log_prob_of_sequence(yw_ids, yw_mask, inject=True)
                lp_vsys_yl = self.model.log_prob_of_sequence(yl_ids, yl_mask, inject=True)

                # ── log p_base(yw) and log p_base(yl) (no grad) ──────────
                with torch.no_grad():
                    lp_base_yw = self.model.log_prob_of_sequence(yw_ids, yw_mask, inject=False)
                    lp_base_yl = self.model.log_prob_of_sequence(yl_ids, yl_mask, inject=False)

                loss = dpo_loss(lp_vsys_yw, lp_base_yw, lp_vsys_yl, lp_base_yl, self.beta)
                loss = loss / self.grad_accum_steps
                loss.backward()
                epoch_loss += loss.item() * self.grad_accum_steps

                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_([self.model.vsys], max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    self._warmup_lr()

                pbar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}"})

            avg_loss = epoch_loss / len(self.loader)
            print(f"  Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

            # Save checkpoint each epoch
            ckpt_path = os.path.join(self.output_dir, f"sysvec_epoch{epoch+1}.pt")
            self.model.save_sysvec(ckpt_path)

        # Save final vector
        final_path = os.path.join(self.output_dir, "sysvec_final.pt")
        self.model.save_sysvec(final_path)
        print(f"\n[SysVecTrainer] Training complete. Final vector saved to {final_path}")
        return final_path

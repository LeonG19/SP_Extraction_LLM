"""
SysVec Trainer
==============
DPO-style optimisation of the vsys vector.
Paper: Section 3.4, eq. 4-5 (Cao et al., CCS '25)

No relative imports — works when all files are in the same flat folder.
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

from model import SysVec          # flat import — same folder


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """
    Holds tokenised (yw, yl) preference pairs.

    yw = full sequence WITH system prompt  (preferred)
    yl = full sequence WITHOUT system prompt (dispreferred)
    """

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad variable-length sequences within a batch."""
    keys = batch[0].keys()
    out = {}
    pad_token_id = batch[0].get("pad_token_id", 0)
    for k in keys:
        values = [item[k] for item in batch]

        # Skip non-tensor entries (e.g. pad_token_id stored as plain int)
        if not isinstance(values[0], torch.Tensor):
            out[k] = values[0]   # same for all items in the batch
            continue

        pad_val = 0 if "mask" in k else pad_token_id
        max_len = max(t.size(0) for t in values)
        padded = torch.stack(
            [F.pad(t, (0, max_len - t.size(0)), value=pad_val) for t in values]
        )
        out[k] = padded
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data synthesis
# ─────────────────────────────────────────────────────────────────────────────

def save_preference_samples(samples: List[Dict], path: str):
    """
    Save (yw, yl) preference pairs to disk as a .pt file.
    Tensors are moved to CPU before saving so they can be reloaded on any device.

    Also writes a human-readable .jsonl sidecar so you can inspect
    the actual text of each pair without loading the full tensor file.
    """
    # Save tensors
    cpu_samples = []
    for s in samples:
        cpu_samples.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in s.items()})
    torch.save(cpu_samples, path)
    print(f"[DPO Data] Saved {len(samples)} pairs -> {path}")


def load_preference_samples(path: str) -> List[Dict]:
    """Load previously saved (yw, yl) pairs from a .pt file."""
    samples = torch.load(path, map_location="cpu")
    print(f"[DPO Data] Loaded {len(samples)} pairs <- {path}")
    return samples


def save_preference_samples_readable(
    samples: List[Dict],
    tokenizer,
    path: str,
):
    """
    Write a human-readable JSONL sidecar so you can inspect what
    the model actually generated for each preference pair.

    Each line is a JSON object:
    {
        "index": 0,
        "yw": "<full decoded preferred sequence>",
        "yl": "<full decoded dispreferred sequence>"
    }
    """
    import json
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            yw_text = tokenizer.decode(s["yw_input_ids"], skip_special_tokens=True)
            yl_text = tokenizer.decode(s["yl_input_ids"], skip_special_tokens=True)
            f.write(json.dumps({
                "index": i,
                "yw":    yw_text,
                "yl":    yl_text,
            }, ensure_ascii=False) + "\n")
    print(f"[DPO Data] Readable sidecar -> {path}")


def build_preference_samples(
    sysvec_model: SysVec,
    system_prompt: str,
    user_questions: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    save_path: Optional[str] = None,
    resume: bool = True,
) -> List[Dict]:
    """
    Generate (yw, yl) pairs using the base model (paper eq. 4):
        yw = f(system_prompt + question)   -- preferred
        yl = f(question alone)             -- dispreferred

    Args:
        sysvec_model:  The SysVec model (base model used for generation).
        system_prompt: The application system prompt text.
        user_questions: List of user question strings.
        max_new_tokens: Max tokens to generate per response.
        batch_size:    Questions to process per GPU batch.
        save_path:     If set, save the generated pairs to this .pt path.
                       A human-readable .jsonl sidecar is also written
                       alongside it at <save_path>.jsonl
        resume:        If True and save_path exists, load from disk instead
                       of regenerating. Set False to force regeneration.
    """
    # ── Resume from disk if available ────────────────────────────────────────
    if save_path and resume and os.path.exists(save_path):
        print(f"[DPO Data] Found existing dataset at {save_path} -- loading.")
        print(f"[DPO Data] Delete the file or pass resume=False to regenerate.")
        return load_preference_samples(save_path)

    tokenizer = sysvec_model.tokenizer
    model     = sysvec_model
    device    = sysvec_model.device
    samples   = []

    total = len(user_questions)
    print(f"[DPO Data] Generating {total} (yw, yl) pairs  "
          f"(batch_size={batch_size}, max_new_tokens={max_new_tokens})")

    for i in range(0, total, batch_size):
        batch_qs = user_questions[i: i + batch_size]

        # --- yw: with system prompt ---
        yw_texts = [_fmt_with_system(tokenizer, system_prompt, q) for q in batch_qs]
        yw_inputs = tokenizer(
            yw_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(device)

        with torch.inference_mode():
            yw_out = model.generate(
                yw_inputs.input_ids,
                attention_mask=yw_inputs.attention_mask,
                inject=False,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # --- yl: without system prompt ---
        yl_texts = [_fmt_without_system(tokenizer, q) for q in batch_qs]
        yl_inputs = tokenizer(
            yl_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(device)

        with torch.inference_mode():
            yl_out = model.generate(
                yl_inputs.input_ids,
                attention_mask=yl_inputs.attention_mask,
                inject=False,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        for j in range(len(batch_qs)):
            yw_ids = yw_out[j].cpu()
            yl_ids = yl_out[j].cpu()
            samples.append({
                "yw_input_ids":      yw_ids,
                "yw_attention_mask": (yw_ids != tokenizer.pad_token_id).long(),
                "yl_input_ids":      yl_ids,
                "yl_attention_mask": (yl_ids != tokenizer.pad_token_id).long(),
                "pad_token_id":      tokenizer.pad_token_id,
            })

        print(f"  Synthesised {min(i + batch_size, total)}/{total} pairs")

    # ── Persist ───────────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_preference_samples(samples, save_path)
        save_preference_samples_readable(
            samples, tokenizer, save_path + ".jsonl"
        )

    return samples


def _fmt_with_system(tokenizer, system_prompt: str, user_query: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_query}\n<|assistant|>\n"


def _fmt_without_system(tokenizer, user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<|user|>\n{user_query}\n<|assistant|>\n"


# ─────────────────────────────────────────────────────────────────────────────
# DPO Loss  (paper eq. 5)
# ─────────────────────────────────────────────────────────────────────────────

def dpo_loss(
    log_prob_vsys_yw: torch.Tensor,
    log_prob_base_yw: torch.Tensor,
    log_prob_vsys_yl: torch.Tensor,
    log_prob_base_yl: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    pi_yw = log_prob_vsys_yw - log_prob_base_yw
    pi_yl = log_prob_vsys_yl - log_prob_base_yl
    logits = beta * (pi_yw - pi_yl)
    return -F.logsigmoid(logits).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SysVecTrainer:
    """
    Trains vsys using the DPO objective (paper §4.1 hyperparameters).

    Only vsys is updated — base model weights are always frozen.
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
        self.model             = sysvec_model
        self.dataset           = dataset
        self.output_dir        = output_dir
        self.beta              = beta
        self.epochs            = epochs
        self.grad_accum_steps  = grad_accum_steps

        os.makedirs(output_dir, exist_ok=True)

        self.loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn,
        )

        self.optimizer = AdamW(
            [self.model.vsys], lr=lr, weight_decay=weight_decay
        )

        total_steps = epochs * math.ceil(len(self.loader) / grad_accum_steps)
        self.scheduler    = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        self.warmup_steps = warmup_steps
        self.global_step  = 0

    def _warmup_lr(self):
        if self.global_step < self.warmup_steps:
            factor = (self.global_step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * factor

    def train(self):
        self.base_model_eval_mode()
        self.model.vsys.requires_grad_(True)

        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        print(f"\n[SysVecTrainer] Starting training for {self.epochs} epochs")
        print(f"  Dataset size    : {len(self.dataset)}")
        print(f"  Injection layer : {self.model.injection_layer}")
        print(f"  Alpha           : {self.model.alpha}\n")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad()

            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for step, batch in enumerate(pbar):
                device  = self.model.device
                yw_ids  = batch["yw_input_ids"].to(device)
                yw_mask = batch["yw_attention_mask"].to(device)
                yl_ids  = batch["yl_input_ids"].to(device)
                yl_mask = batch["yl_attention_mask"].to(device)

                self.model.train()
                lp_vsys_yw = self.model.log_prob_of_sequence(yw_ids, yw_mask, inject=True)
                lp_vsys_yl = self.model.log_prob_of_sequence(yl_ids, yl_mask, inject=True)

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

            avg = epoch_loss / len(self.loader)
            print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")
            self.model.save_sysvec(
                os.path.join(self.output_dir, f"sysvec_epoch{epoch+1}.pt")
            )

        final = os.path.join(self.output_dir, "sysvec_final.pt")
        self.model.save_sysvec(final)
        print(f"\n[SysVecTrainer] Done. Final vector -> {final}")
        return final

    def base_model_eval_mode(self):
        self.model.base_model.eval()
        for p in self.model.base_model.parameters():
            p.requires_grad_(False)
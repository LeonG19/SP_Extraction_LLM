"""
SysVec: Activation Steering for System Prompt Protection
Paper: "You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors"
(Cao et al., CCS '25, arXiv:2509.21884)

Core model wrapper that:
1. Injects a learned representation vector vsys at layer l during forward passes
2. Exposes optimization hooks for DPO-style training
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class SysVec(nn.Module):
    """
    Wraps any HuggingFace CausalLM and injects a trained system vector
    into a chosen hidden-state layer at inference and training time.

    The forward pass becomes (paper eq. 3):
        f(x, vsys) = f^{l+1:L}( f^{1:l}(x) + alpha * vsys )

    Args:
        model_name_or_path: HuggingFace model id or local path.
        injection_layer:    Layer index l at which vsys is added.
                            Paper defaults: Llama-2/3 → 15, Mistral → 13.
        alpha:              Steering strength scalar.
                            Paper defaults: Llama-2/3 → 1.0, Mistral → 2.5.
        device:             torch device string.
        torch_dtype:        dtype for model weights (default bfloat16).
    """

    def __init__(
        self,
        model_name_or_path: str,
        injection_layer: int = 15,
        alpha: float = 1.0,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()

        self.injection_layer = injection_layer
        self.alpha = alpha
        self.device = device

        # ── Load base model & tokenizer ──────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.base_model.eval()

        hidden_size = self.base_model.config.hidden_size

        # ── Learnable system vector (shape: [1, hidden_size]) ────────────
        # Initialised to zero so the model starts identical to the base.
        self.vsys = nn.Parameter(
            torch.zeros(1, hidden_size, dtype=torch_dtype, device=device)
        )

        # ── Hook handle storage ──────────────────────────────────────────
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._injection_active: bool = False

    # ────────────────────────────────────────────────────────────────────
    # Hook helpers
    # ────────────────────────────────────────────────────────────────────

    def _get_decoder_layers(self):
        """Return the list of transformer decoder blocks for common architectures."""
        cfg_type = type(self.base_model.config).__name__.lower()
        model = self.base_model

        if hasattr(model, "model"):               # Llama / Mistral / Gemma
            return model.model.layers
        if hasattr(model, "transformer"):          # GPT-2 / Falcon style
            if hasattr(model.transformer, "h"):
                return model.transformer.h
            if hasattr(model.transformer, "blocks"):
                return model.transformer.blocks
        raise ValueError(
            f"Cannot locate decoder layers for model type '{cfg_type}'. "
            "Override _get_decoder_layers() for your architecture."
        )

    def _make_hook(self):
        """
        Returns a forward hook that adds alpha*vsys to the OUTPUT hidden state
        of the chosen layer.  The hook fires only when self._injection_active.
        """
        vsys = self.vsys
        alpha = self.alpha

        def hook(module, input, output):
            if not self._injection_active:
                return output

            # output is a tuple; index 0 is the hidden state tensor [B, T, H]
            hidden = output[0]
            # Broadcast vsys across batch and sequence dimensions
            hidden = hidden + alpha * vsys.to(hidden.dtype)
            return (hidden,) + output[1:]

        return hook

    def register_injection_hook(self):
        """Attach the activation-addition hook to layer l."""
        if self._hook_handle is not None:
            return  # already registered

        layers = self._get_decoder_layers()
        target_layer = layers[self.injection_layer]
        self._hook_handle = target_layer.register_forward_hook(self._make_hook())

    def remove_injection_hook(self):
        """Detach the hook (e.g., for base-model comparisons)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask=None, labels=None, inject=True, **kwargs):
        """
        Run the (optionally injected) model.

        Args:
            input_ids:       [B, T] token ids.
            attention_mask:  [B, T] mask.
            labels:          [B, T] for language-model loss (optional).
            inject:          Whether to add vsys during this forward pass.
        """
        self.register_injection_hook()
        self._injection_active = inject

        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        self._injection_active = False
        return output

    # ────────────────────────────────────────────────────────────────────
    # Convenience: log-probabilities of a sequence
    # ────────────────────────────────────────────────────────────────────

    def log_prob_of_sequence(self, input_ids, attention_mask, inject: bool) -> torch.Tensor:
        """
        Compute the sum of per-token log-probabilities for a full sequence.
        Used in the DPO objective.

        Returns:
            Scalar tensor: sum_t log p(t | t<t, inject=inject).
        """
        with torch.set_grad_enabled(self.training and inject):
            out = self.forward(input_ids, attention_mask, inject=inject)

        logits = out.logits  # [B, T, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [B, T-1]

        # Mask padding
        if attention_mask is not None:
            mask = attention_mask[:, 1:].float()
            token_log_probs = token_log_probs * mask

        return token_log_probs.sum(dim=-1)  # [B]

    # ────────────────────────────────────────────────────────────────────
    # Generation (inference)
    # ────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, inject=True, **gen_kwargs):
        """Generate tokens with optional vsys injection."""
        self.register_injection_hook()
        self._injection_active = inject

        out = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )

        self._injection_active = False
        return out

    # ────────────────────────────────────────────────────────────────────
    # Save / load vsys only
    # ────────────────────────────────────────────────────────────────────

    def save_sysvec(self, path: str):
        """Persist only the learned vector (not the full model weights)."""
        torch.save(
            {
                "vsys": self.vsys.data,
                "injection_layer": self.injection_layer,
                "alpha": self.alpha,
            },
            path,
        )
        print(f"[SysVec] Saved system vector to {path}")

    def load_sysvec(self, path: str):
        """Load a previously saved vector into this instance."""
        ckpt = torch.load(path, map_location=self.device)
        self.vsys.data = ckpt["vsys"].to(self.device)
        self.injection_layer = ckpt["injection_layer"]
        self.alpha = ckpt["alpha"]
        # Re-register hook in case layer changed
        self.remove_injection_hook()
        self.register_injection_hook()
        print(f"[SysVec] Loaded system vector from {path}")

"""
SysVec: Activation Steering for System Prompt Protection
Paper: "You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors"
(Cao et al., CCS '25, arXiv:2509.21884)

Hook strategy (v3)
------------------
Previous versions hooked the decoder LAYER output and tried to reconstruct
the output tuple/dataclass. This breaks because Transformers internally
unpacks the layer output and passes only the hidden state tensor to the
next layer's input_layernorm — so returning a tuple where a tensor is
expected causes "AttributeError: 'tuple' object has no attribute 'dtype'".

Fix: use register_forward_hook on the INPUT LAYERNORM of layer l+1 instead.
The layernorm receives the hidden state as a plain tensor, we add vsys to it,
and return the modified tensor. This is simpler, version-agnostic, and
guaranteed to work regardless of how the decoder layer packages its outputs.

Alternative used when l+1 does not exist (last layer): hook the output
projection of layer l (the final MLP or the layer itself via a pre-hook
on layer l+1's input).

Actually the cleanest cross-version approach is a register_forward_pre_hook
on layer l+1 — it receives the INPUT to that layer, which is always a plain
tensor tuple (hidden_states, attention_mask, ...). We modify hidden_states
and return the modified args. This works on all Transformers versions.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class SysVec(nn.Module):
    """
    Wraps any HuggingFace CausalLM and injects a trained system vector
    into a chosen hidden-state layer at inference and training time.

    Forward pass (paper eq. 3):
        f(x, vsys) = f^{l+1:L}( f^{1:l}(x) + alpha * vsys )

    We implement this by hooking the INPUT of layer l+1 (pre-hook),
    which receives the hidden state as a plain tensor -- version-agnostic
    and works with all Transformers output formats.

    Args:
        model_name_or_path: HuggingFace model id or local path.
        injection_layer:    Layer l after which vsys is injected.
                            Paper: Llama-2/3 -> 15, Mistral -> 13.
        alpha:              Steering strength scalar.
                            Paper: Llama-2/3 -> 1.0, Mistral -> 2.5.
        device:             torch device string.
        torch_dtype:        Weight dtype (default bfloat16).
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

        # ── Tokenizer ────────────────────────────────────────────────────────
        # padding_side='left' is mandatory for decoder-only models.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Base model ───────────────────────────────────────────────────────
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.base_model.eval()

        hidden_size = self.base_model.config.hidden_size

        # ── Learnable system vector ──────────────────────────────────────────
        # Shape [1, hidden_size]. Zero-init -> base model at step 0.
        # THE ONLY parameter updated during training.
        self.vsys = nn.Parameter(
            torch.zeros(1, hidden_size, dtype=torch_dtype, device=device)
        )

        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._injection_active: bool = False

    # ── Layer discovery ───────────────────────────────────────────────────────

    def _get_decoder_layers(self):
        """
        Return the ModuleList of decoder blocks.
        Covers: Llama, Mistral, Gemma, Qwen2, Phi-3, GPT-2, Falcon, OPT.
        """
        m = self.base_model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers                    # Llama / Mistral / Gemma / Qwen2 / Phi
        if hasattr(m, "transformer"):
            if hasattr(m.transformer, "h"):
                return m.transformer.h               # GPT-2 / Falcon
            if hasattr(m.transformer, "blocks"):
                return m.transformer.blocks
        if hasattr(m, "model") and hasattr(m.model, "decoder"):
            return m.model.decoder.layers            # OPT
        raise ValueError(
            f"Cannot locate decoder layers for '{type(m).__name__}'. "
            "Add a branch to SysVec._get_decoder_layers()."
        )

    # ── Hook ─────────────────────────────────────────────────────────────────

    def _make_pre_hook(self):
        """
        Build a forward PRE-hook for layer (injection_layer + 1).

        A pre-hook receives the INPUT arguments to the module before its
        forward() runs. For decoder layers the first positional arg is always
        the hidden state tensor [B, T, H] -- a plain tensor regardless of
        Transformers version. We add vsys to it and return the modified args.

        This is simpler and more robust than hooking the OUTPUT of layer l,
        which changed format across Transformers versions (tuple -> dataclass).

        If injection_layer is the last layer we fall back to hooking the
        output of the final norm instead (see register_injection_hook).
        """
        model_ref = self

        def pre_hook(module, args):
            if not model_ref._injection_active:
                return args

            # args[0] is always the hidden state tensor [B, T, H]
            hidden  = args[0]
            steered = hidden + model_ref.alpha * model_ref.vsys.to(hidden.dtype)

            # Return the full args tuple with the first element replaced
            return (steered,) + args[1:]

        return pre_hook

    def register_injection_hook(self):
        """
        Attach the pre-hook to the INPUT of layer (injection_layer + 1).
        This is idempotent -- safe to call multiple times.
        """
        if self._hook_handle is not None:
            return

        layers = self._get_decoder_layers()
        n_layers = len(layers)

        next_layer_idx = self.injection_layer + 1

        if next_layer_idx < n_layers:
            # Normal case: hook the input of the next decoder layer
            target = layers[next_layer_idx]
            self._hook_handle = target.register_forward_pre_hook(
                self._make_pre_hook()
            )
        else:
            # Edge case: injection_layer is the last decoder layer.
            # Hook the final layernorm input instead.
            final_norm = self._get_final_norm()
            if final_norm is not None:
                self._hook_handle = final_norm.register_forward_pre_hook(
                    self._make_pre_hook()
                )
            else:
                raise ValueError(
                    f"injection_layer={self.injection_layer} is the last layer "
                    "and no final norm was found. Choose a lower injection_layer."
                )

    def _get_final_norm(self):
        """Return the final layer norm of the model (after all decoder layers)."""
        m = self.base_model
        # Llama / Mistral / Gemma
        if hasattr(m, "model") and hasattr(m.model, "norm"):
            return m.model.norm
        # GPT-2
        if hasattr(m, "transformer") and hasattr(m.transformer, "ln_f"):
            return m.transformer.ln_f
        # OPT
        if hasattr(m, "model") and hasattr(m.model, "decoder") \
                and hasattr(m.model.decoder, "final_layer_norm"):
            return m.model.decoder.final_layer_norm
        return None

    def remove_injection_hook(self):
        """Detach the hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask=None, labels=None,
                inject=True, **kwargs):
        """
        Forward pass with optional vsys injection.
        try/finally ensures _injection_active is always reset.
        """
        self.register_injection_hook()
        self._injection_active = inject
        try:
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            self._injection_active = False
        return output

    # ── Log-probabilities ─────────────────────────────────────────────────────

    def log_prob_of_sequence(
        self, input_ids, attention_mask, inject: bool
    ) -> torch.Tensor:
        """
        Sum of per-token log-probabilities for a complete sequence.
        Used by the DPO loss for both p_vsys(y) and p_base(y).
        """
        with torch.set_grad_enabled(self.training and inject):
            out = self.forward(input_ids, attention_mask, inject=inject)

        logits        = out.logits                        # [B, T, V]
        shift_logits  = logits[:, :-1, :].contiguous()
        shift_labels  = input_ids[:, 1:].contiguous()

        log_probs       = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)                                      # [B, T-1]

        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask[:, 1:].float()

        return token_log_probs.sum(dim=-1)                 # [B]

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, inject=True, **gen_kwargs):
        """Generate tokens with optional vsys injection."""
        self.register_injection_hook()
        self._injection_active = inject
        try:
            out = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
        finally:
            self._injection_active = False
        return out

    # ── Save / load ───────────────────────────────────────────────────────────

    def save_sysvec(self, path: str):
        torch.save(
            {
                "vsys":            self.vsys.data,
                "injection_layer": self.injection_layer,
                "alpha":           self.alpha,
            },
            path,
        )
        print(f"[SysVec] Saved -> {path}")

    def load_sysvec(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.vsys.data       = ckpt["vsys"].to(self.device)
        self.injection_layer = ckpt["injection_layer"]
        self.alpha           = ckpt["alpha"]
        self.remove_injection_hook()
        self.register_injection_hook()
        print(f"[SysVec] Loaded <- {path}")
from typing import Callable, Optional

import torch
import torch.optim as optim


ALIGNMENT_EPS = 1e-8


class TCG(optim.Optimizer):
    """Temporal Consistency Gating wrapper for a base DP optimizer.

    TCG computes a coordinate-wise gate from temporal histories of privatized
    gradients and applies that gate to the update produced by the base
    optimizer. The wrapper assumes ``p.grad`` is already the privatized gradient
    provided by the DP training pipeline.
    """

    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        beta_fast: float = 0.8,
        beta_slow: float = 0.98,
        gate_type: str = "hard",
        gate_tau: float = 0.0,
    ):
        if not (0.0 <= beta_fast < beta_slow < 1.0):
            raise ValueError("Expected 0 <= beta_fast < beta_slow < 1")
        if gate_type not in {"hard", "soft"}:
            raise ValueError("gate_type must be either 'hard' or 'soft'")
        if gate_type == "soft" and gate_tau <= 0.0:
            raise ValueError("gate_tau must be > 0")

        self.base_optimizer = base_optimizer
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.gate_type = gate_type
        self.gate_tau = gate_tau

        super().__init__(self.base_optimizer.param_groups, defaults={})

        self.current_gate_mean = 1.0
        self.current_consistency_ratio = 1.0
        self.current_support_ratio = 1.0
        self.current_score_mean = 0.0

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "wrapper_state": super().state_dict(),
            "base_optimizer_state": self.base_optimizer.state_dict(),
            "wrapper_hparams": {
                "beta_fast": self.beta_fast,
                "beta_slow": self.beta_slow,
                "gate_type": self.gate_type,
                "gate_tau": self.gate_tau,
            },
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["wrapper_state"])
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])

        hparams = state_dict.get("wrapper_hparams", {})
        self.beta_fast = hparams.get("beta_fast", self.beta_fast)
        self.beta_slow = hparams.get("beta_slow", self.beta_slow)
        self.gate_type = hparams.get("gate_type", self.gate_type)
        self.gate_tau = hparams.get("gate_tau", self.gate_tau)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        old_params = {}
        gates = {}
        next_states = {}

        total_elems = 0
        gate_sum = 0.0
        consistency_sum = 0.0
        support_sum = 0.0
        score_sum = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                privatized_grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m_fast"] = torch.zeros_like(p)
                    state["m_slow"] = torch.zeros_like(p)

                prev_fast = state["m_fast"]
                prev_slow = state["m_slow"]
                consistency_score = (prev_fast * prev_slow) / (
                    prev_fast.abs() * prev_slow.abs() + ALIGNMENT_EPS
                )
                support_score = (privatized_grad * prev_slow) / (
                    privatized_grad.abs() * prev_slow.abs() + ALIGNMENT_EPS
                )
                score = torch.maximum(consistency_score, support_score)

                consistency = (consistency_score > 0).to(
                    privatized_grad.dtype
                )
                support = (support_score > 0).to(privatized_grad.dtype)

                if self.gate_type == "hard":
                    gate = (score > 0).to(privatized_grad.dtype)
                else:
                    gate = torch.sigmoid(score / self.gate_tau)

                old_params[p] = p.data.clone()
                gates[p] = gate
                next_states[p] = {
                    "m_fast": (
                        self.beta_fast * prev_fast
                        + (1.0 - self.beta_fast) * privatized_grad
                    ).detach(),
                    "m_slow": (
                        self.beta_slow * prev_slow
                        + (1.0 - self.beta_slow) * privatized_grad
                    ).detach(),
                }

                total_elems += gate.numel()
                gate_sum += gate.sum().item()
                consistency_sum += consistency.sum().item()
                support_sum += support.sum().item()
                score_sum += score.sum().item()

        base_out = (
            self.base_optimizer.step(closure)
            if closure is not None
            else self.base_optimizer.step()
        )

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in old_params:
                    continue

                base_update = p.data - old_params[p]
                gated_update = gates[p] * base_update
                p.data.copy_(old_params[p] + gated_update)

                state = self.state[p]
                state["m_fast"] = next_states[p]["m_fast"]
                state["m_slow"] = next_states[p]["m_slow"]

        if total_elems > 0:
            self.current_gate_mean = gate_sum / total_elems
            self.current_consistency_ratio = consistency_sum / total_elems
            self.current_support_ratio = support_sum / total_elems
            self.current_score_mean = score_sum / total_elems
        else:
            self.current_gate_mean = 1.0
            self.current_consistency_ratio = 1.0
            self.current_support_ratio = 1.0
            self.current_score_mean = 0.0

        return base_out

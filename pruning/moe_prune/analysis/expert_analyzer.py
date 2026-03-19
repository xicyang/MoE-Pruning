"""Expert analyzer wrapper for MoE layers."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ExpertAnalyzerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.enable_recording = False

        self.router_probs_sum = None
        self.expert_norms_sum = None
        self.sample_count = 0

        self.all_inputs = []
        self.all_outputs = []
        
        self.num_experts = self._resolve_num_experts()
        self.top_k = self._resolve_top_k()
        self._normalize_topk_probs = self._resolve_norm_topk_flag()

    def _resolve_num_experts(self) -> int:
        if hasattr(self.model, "num_experts") and self.model.num_experts:
            return int(self.model.num_experts)
        if hasattr(self.model, "num_local_experts") and self.model.num_local_experts:
            return int(self.model.num_local_experts)
        if hasattr(self.model, "experts"):
            return len(self.model.experts)
        raise ValueError("Cannot determine number of experts for wrapped MoE block.")

    def _resolve_top_k(self) -> int:
        for attr in ("top_k", "num_selected_experts", "num_experts_per_tok", "num_experts_per_token"):
            if hasattr(self.model, attr):
                value = getattr(self.model, attr)
                if value:
                    return int(value)
        raise ValueError("Cannot determine top-k routing setting for wrapped MoE block.")

    def _resolve_norm_topk_flag(self) -> bool:
        if hasattr(self.model, "norm_topk_prob"):
            return bool(self.model.norm_topk_prob)
        if hasattr(self.model, "normalize_topk_prob"):
            return bool(self.model.normalize_topk_prob)
        gate = getattr(self.model, "gate", None)
        if gate is not None:
            if hasattr(gate, "norm_topk_prob"):
                return bool(getattr(gate, "norm_topk_prob"))
            if hasattr(gate, "normalize_topk_prob"):
                return bool(getattr(gate, "normalize_topk_prob"))
        return True

    def _compute_router_logits(self, hidden_states_flat: torch.Tensor) -> torch.Tensor:
        gate = getattr(self.model, "gate", None)
        if gate is None:
            raise ValueError("MoE block missing gate module.")
        bias: Optional[torch.nn.Parameter] = getattr(gate, "bias", None)
        if isinstance(gate, torch.nn.Linear):
            return gate(hidden_states_flat)
        weight = getattr(gate, "weight", None)
        if weight is None:
            raise ValueError("Unsupported gate type: cannot find weight parameter.")
        return F.linear(hidden_states_flat, weight, bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self._compute_router_logits(hidden_states_flat)

        if self.enable_recording:
            self.all_inputs.append(hidden_states.detach())

        routing_weights_all = F.softmax(router_logits, dim=1, dtype=torch.float)

        routing_weights, selected_experts = torch.topk(routing_weights_all, self.top_k, dim=-1)
        if self._normalize_topk_probs:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] > 0:
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                current_state = hidden_states_flat[None, top_x_list].reshape(-1, hidden_dim)
                current_routing_weights = routing_weights[top_x_list, idx_list, None]
                expert_output = expert_layer(current_state, current_routing_weights)
                final_hidden_states.index_add_(0, top_x, expert_output.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.enable_recording:
            self.all_outputs.append(final_hidden_states.detach())

            num_samples = batch_size * sequence_length

            if self.router_probs_sum is None:
                self.router_probs_sum = torch.zeros(self.num_experts, device=hidden_states.device)
                self.expert_norms_sum = torch.zeros(self.num_experts, device=hidden_states.device)

            router_probs = F.softmax(router_logits, dim=1)
            router_probs_batch_sum = router_probs.sum(dim=0)
            self.router_probs_sum += router_probs_batch_sum

            with torch.inference_mode():
                ones_routing = torch.ones(
                    (hidden_states_flat.shape[0], 1),
                    device=hidden_states_flat.device,
                    dtype=hidden_states_flat.dtype,
                )

                for expert_idx in range(self.num_experts):
                    expert_layer = self.model.experts[expert_idx]
                    expert_output_all = expert_layer(hidden_states_flat.detach(), ones_routing)
                    expert_output_norms = torch.norm(expert_output_all, dim=1)
                    self.expert_norms_sum[expert_idx] += expert_output_norms.sum()

            self.sample_count += num_samples

        return final_hidden_states, router_logits

    @torch.no_grad()
    def compute_expert_loss(self) -> Dict[int, float]:
        if not self.enable_recording:
            raise ValueError(f"Recording not enabled. enable_recording={self.enable_recording}")
        if len(self.all_inputs) == 0:
            raise ValueError("No input data recorded")
        if len(self.all_inputs) != len(self.all_outputs):
            raise ValueError("Input and output count mismatch")

        loss_scores = {}
        num_experts = self.num_experts
        num_batches = len(self.all_inputs)

        for expert_idx in range(num_experts):
            total_loss = 0.0

            for input_hidden, original_output in zip(self.all_inputs, self.all_outputs):
                original_output = original_output.to(dtype=torch.float64)
                expert_output_single, _ = self._forward_with_single_expert(input_hidden, expert_idx)
                expert_output_single = expert_output_single.to(dtype=torch.float64)
                loss = torch.norm(original_output - expert_output_single).item()
                total_loss += loss

            loss_scores[expert_idx] = total_loss / num_batches

        return loss_scores

    def _forward_with_single_expert(
        self, hidden_states: torch.Tensor, expert_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self._compute_router_logits(hidden_states_flat)
        router_logits_masked = router_logits.clone()
        for e in range(self.num_experts):
            if e != expert_idx:
                router_logits_masked[:, e] = -float("inf")

        routing_weights = F.softmax(router_logits_masked, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self._normalize_topk_probs:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_layer = self.model.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        if top_x.shape[0] > 0:
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states_flat[None, top_x_list].reshape(-1, hidden_dim)
            current_routing_weights = routing_weights[top_x_list, idx_list, None]
            current_hidden_states = expert_layer(current_state, current_routing_weights)
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    def clear_records(self):
        self.router_probs_sum = None
        self.expert_norms_sum = None
        self.sample_count = 0
        self.all_inputs = []
        self.all_outputs = []

    def get_statistics(self) -> Dict:
        if self.sample_count == 0:
            raise ValueError("No data recorded")

        router_probs_mean = self.router_probs_sum / self.sample_count
        expert_norms_mean = self.expert_norms_sum / self.sample_count

        return {
            "router_probs_mean": router_probs_mean,
            "expert_norms_mean": expert_norms_mean,
            "sample_count": self.sample_count,
        }
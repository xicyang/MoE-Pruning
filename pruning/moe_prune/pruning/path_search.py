"""Frequency-based path search with per-layer outlier filtering (v4)."""

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from moe_prune.config import PathSearchConfig
from moe_prune.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PathSearchResult:
    """Result of path search step."""

    mask_path: Path


def percentile(sorted_values: List[float], pct: float) -> float:
    """Compute percentile (0-1) for pre-sorted values."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 1:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * pct
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_values[int(idx)]
    fraction = idx - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def compute_iqr_bounds(values: List[float], multiplier: float) -> Optional[Tuple[float, float]]:
    """Return (lower_bound, upper_bound) using Tukey IQR rule."""
    if len(values) < 4:
        return None
    sorted_vals = sorted(values)
    q1 = percentile(sorted_vals, 0.25)
    q3 = percentile(sorted_vals, 0.75)
    iqr = q3 - q1
    if iqr <= 1e-12:
        return None
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


@dataclass
class OutlierConfig:
    """Configuration for the outlier detector."""

    loss_iqr_multiplier: float = 3
    norm_iqr_multiplier: float = 3
    min_values: int = 4


@dataclass
class LayerOutlierResult:
    """Holds force-keep/remove decisions for a single layer."""

    layer_idx: int
    force_keep: Set[int] = field(default_factory=set)
    force_remove: Set[int] = field(default_factory=set)
    reasons: Dict[int, List[str]] = field(default_factory=lambda: defaultdict(list))
    loss_bounds: Optional[Tuple[float, float]] = None
    norm_bounds: Optional[Tuple[float, float]] = None


class LayerOutlierInspector:
    """Detect outliers per layer for loss and expert norms."""

    def __init__(self, config: OutlierConfig):
        self.config = config

    def analyze_layer(self, layer_idx: int, layer_data: Dict) -> LayerOutlierResult:
        result = LayerOutlierResult(layer_idx=layer_idx)
        loss_values = []
        expert_norms = layer_data["expert_norms_mean"]
        num_experts = len(expert_norms)

        for expert_idx in range(num_experts):
            loss_values.append(layer_data["loss"][f"expert_{expert_idx}"])

        if len(loss_values) >= self.config.min_values:
            result.loss_bounds = compute_iqr_bounds(loss_values, self.config.loss_iqr_multiplier)

        if len(expert_norms) >= self.config.min_values:
            result.norm_bounds = compute_iqr_bounds(expert_norms, self.config.norm_iqr_multiplier)

        def mark_keep(idx: int, reason: str):
            result.force_keep.add(idx)
            result.reasons[idx].append(reason)

        def mark_remove(idx: int, reason: str):
            result.force_remove.add(idx)
            result.reasons[idx].append(reason)

        if result.loss_bounds is not None:
            lower, upper = result.loss_bounds
            for expert_idx, loss_value in enumerate(loss_values):
                if loss_value < lower:
                    mark_keep(expert_idx, f"loss({loss_value:.5f}) < {lower:.5f}")
                elif loss_value > upper:
                    mark_remove(expert_idx, f"loss({loss_value:.5f}) > {upper:.5f}")

        if result.norm_bounds is not None:
            lower, upper = result.norm_bounds
            for expert_idx, norm_value in enumerate(expert_norms):
                if norm_value > upper:
                    mark_keep(expert_idx, f"norm({norm_value:.5f}) > {upper:.5f}")
                elif norm_value < lower:
                    mark_remove(expert_idx, f"norm({norm_value:.5f}) < {lower:.5f}")

        conflicted = result.force_keep & result.force_remove
        if conflicted:
            for expert_idx in conflicted:
                result.force_remove.discard(expert_idx)
                result.reasons[expert_idx].append("conflict -> prefer keep")

        return result


@dataclass(order=True)
class PathEntry:
    log_weight: float
    node_id: str = field(compare=False)
    predecessor: Optional[Tuple[str, int]] = field(compare=False)
    path_experts: Set[Tuple[int, int]] = field(compare=False, default_factory=set)


class MoEGraph:
    """Graph structure for MoE model with routers and experts."""

    def __init__(
        self,
        analysis_results: Dict,
        tau: float = 1.0,
        initial_disabled: Optional[Set[Tuple[int, int]]] = None,
    ):
        self.analysis_results = analysis_results
        self.num_layers = len([k for k in analysis_results.keys() if k.startswith("layer_")])
        # Dynamically determine num_experts from analysis results
        # Try to get from first layer's expert_norms_mean or router_probs_mean
        if self.num_layers > 0:
            first_layer_key = f"layer_0"
            if first_layer_key in analysis_results:
                layer_data = analysis_results[first_layer_key]
                if "expert_norms_mean" in layer_data:
                    self.num_experts = len(layer_data["expert_norms_mean"])
                elif "router_probs_mean" in layer_data:
                    self.num_experts = len(layer_data["router_probs_mean"])
                else:
                    raise ValueError(f"Cannot determine num_experts from {first_layer_key}")
            else:
                raise ValueError(f"Layer 0 not found in analysis_results")
        else:
            raise ValueError("No layers found in analysis_results")
        self.tau = tau

        self.nodes = self._construct_nodes()
        self.disabled_experts: Set[Tuple[int, int]] = set(initial_disabled or set())
        self.edges = self._construct_edges()
        self.node_weights = self._compute_node_weights()
        self.topological_order = self._compute_topological_order()

    def _construct_nodes(self) -> List[str]:
        nodes = ["source"]
        for layer_idx in range(self.num_layers):
            nodes.append(f"router_{layer_idx}")
            for expert_idx in range(self.num_experts):
                nodes.append(f"expert_{layer_idx}_{expert_idx}")
        nodes.append("sink")
        return nodes

    def _construct_edges(self) -> Dict[str, List[Tuple[str, float]]]:
        edges = defaultdict(list)
        edges["source"].append(("router_0", 1.0))

        for layer_idx in range(self.num_layers):
            layer_key = f"layer_{layer_idx}"
            layer_data = self.analysis_results[layer_key]
            router_node = f"router_{layer_idx}"
            router_probs = layer_data["router_probs_mean"]
            expert_norms = layer_data["expert_norms_mean"]

            # 与 path_searchv4.py 保持一致：直接使用 router_probs，不归一化
            for expert_idx in range(self.num_experts):
                if not self.is_expert_enabled(layer_idx, expert_idx):
                    continue
                expert_node = f"expert_{layer_idx}_{expert_idx}"
                edges[router_node].append((expert_node, router_probs[expert_idx]))

            next_router = f"router_{layer_idx + 1}" if layer_idx < self.num_layers - 1 else "sink"
            for expert_idx in range(self.num_experts):
                if not self.is_expert_enabled(layer_idx, expert_idx):
                    continue
                expert_node = f"expert_{layer_idx}_{expert_idx}"
                edges[expert_node].append((next_router, expert_norms[expert_idx]))

        return edges

    def _compute_node_weights(self) -> Dict[str, float]:
        node_weights = {}
        for layer_idx in range(self.num_layers):
            layer_key = f"layer_{layer_idx}"
            layer_data = self.analysis_results[layer_key]
            enabled_experts = [
                expert_idx
                for expert_idx in range(self.num_experts)
                if self.is_expert_enabled(layer_idx, expert_idx)
            ]
            disabled_experts = [
                expert_idx
                for expert_idx in range(self.num_experts)
                if not self.is_expert_enabled(layer_idx, expert_idx)
            ]

            if enabled_experts:
                # 与 path_searchv4.py 保持一致：直接计算 exp(-loss/tau)
                exp_terms = []
                for expert_idx in enabled_experts:
                    loss_value = layer_data["loss"][f"expert_{expert_idx}"]
                    exp_terms.append(math.exp(-loss_value / self.tau))

                sum_exp = sum(exp_terms)
                if sum_exp < 1e-10:
                    sum_exp = 1e-10

                for pos, expert_idx in enumerate(enabled_experts):
                    expert_node = f"expert_{layer_idx}_{expert_idx}"
                    node_weights[expert_node] = exp_terms[pos] / sum_exp

            for expert_idx in disabled_experts:
                expert_node = f"expert_{layer_idx}_{expert_idx}"
                node_weights[expert_node] = 0.0

        for node in self.nodes:
            if node not in node_weights:
                node_weights[node] = 1.0

        return node_weights

    def _compute_topological_order(self) -> List[str]:
        order = ["source"]
        for layer_idx in range(self.num_layers):
            order.append(f"router_{layer_idx}")
            for expert_idx in range(self.num_experts):
                order.append(f"expert_{layer_idx}_{expert_idx}")
        order.append("sink")
        return order

    def disable_experts(self, experts: Set[Tuple[int, int]]):
        # 与 path_searchv4.py 保持一致：只更新 node_weights，不更新 edges
        newly_disabled = experts - self.disabled_experts
        self.disabled_experts.update(experts)
        for layer_idx, expert_idx in newly_disabled:
            node = f"expert_{layer_idx}_{expert_idx}"
            self.node_weights[node] = 0.0

    def is_expert_enabled(self, layer_idx: int, expert_idx: int) -> bool:
        return (layer_idx, expert_idx) not in self.disabled_experts

    def get_active_edges(self) -> Dict[str, List[Tuple[str, float]]]:
        active_edges = defaultdict(list)
        if "source" in self.edges:
            active_edges["source"].extend(self.edges["source"])

        for layer_idx in range(self.num_layers):
            router_node = f"router_{layer_idx}"
            layer_key = f"layer_{layer_idx}"
            layer_data = self.analysis_results[layer_key]
            router_probs = layer_data["router_probs_mean"]

            enabled_experts = [
                expert_idx
                for expert_idx in range(self.num_experts)
                if self.is_expert_enabled(layer_idx, expert_idx)
            ]

            if enabled_experts and router_node in self.edges:
                prob_sum = sum(router_probs[idx] for idx in enabled_experts)
                if prob_sum > 0.0:
                    for expert_idx in enabled_experts:
                        neighbor = f"expert_{layer_idx}_{expert_idx}"
                        normalized_prob = router_probs[expert_idx] / prob_sum
                        active_edges[router_node].append((neighbor, normalized_prob))
                else:
                    uniform_prob = 1.0 / len(enabled_experts)
                    for expert_idx in enabled_experts:
                        neighbor = f"expert_{layer_idx}_{expert_idx}"
                        active_edges[router_node].append((neighbor, uniform_prob))

            for expert_idx in enabled_experts:
                expert_node = f"expert_{layer_idx}_{expert_idx}"
                if expert_node in self.edges:
                    active_edges[expert_node].extend(self.edges[expert_node])

        return active_edges


class TopKPathFinder:
    """Top-k path finder allowing overlap, respecting disabled experts."""

    def __init__(self, graph: MoEGraph, k: int):
        self.graph = graph
        self.k = k
        self.H: Dict[str, List[PathEntry]] = {}

    def find_top_k_paths(self) -> List[PathEntry]:
        for node in self.graph.nodes:
            self.H[node] = []

        self.H["source"] = [
            PathEntry(log_weight=0.0, node_id="source", predecessor=None, path_experts=set())
        ]

        for node_v in self.graph.topological_order:
            if not self.H[node_v]:
                continue

            if node_v.startswith("expert_"):
                parts = node_v.split("_")
                layer_idx = int(parts[1])
                expert_idx = int(parts[2])
                if not self.graph.is_expert_enabled(layer_idx, expert_idx):
                    continue

            if node_v not in self.graph.edges:
                continue

            for neighbor_u, edge_weight in self.graph.edges[node_v]:
                temp_entries: List[PathEntry] = []
                for idx, entry in enumerate(self.H[node_v]):
                    log_edge_weight = math.log(edge_weight + 1e-10)
                    log_node_weight = 0.0
                    new_path_experts = entry.path_experts.copy()

                    if neighbor_u.startswith("expert_"):
                        parts = neighbor_u.split("_")
                        layer_idx = int(parts[1])
                        expert_idx = int(parts[2])
                        if not self.graph.is_expert_enabled(layer_idx, expert_idx):
                            continue
                        node_weight = self.graph.node_weights[neighbor_u]
                        # node_weight 已经通过 softmax 归一化到 [0, 1] 范围
                        # 使用 log(weight + 1e-10) 是安全的，因为权重在合理范围内
                        log_node_weight = math.log(node_weight + 1e-10)
                        new_path_experts.add((layer_idx, expert_idx))

                    new_entry = PathEntry(
                        log_weight=entry.log_weight + log_edge_weight + log_node_weight,
                        node_id=neighbor_u,
                        predecessor=(node_v, idx),
                        path_experts=new_path_experts,
                    )
                    temp_entries.append(new_entry)

                if not temp_entries:
                    continue

                self.H[neighbor_u].extend(temp_entries)
                self.H[neighbor_u].sort(key=lambda e: e.log_weight, reverse=True)
                self.H[neighbor_u] = self.H[neighbor_u][: self.k]

        return self.H["sink"][: self.k]


def select_top_paths(graph: MoEGraph, topk: int, unique_paths: bool) -> List[PathEntry]:
    """Return top paths with optional uniqueness constraint."""
    if not unique_paths:
        finder = TopKPathFinder(graph, topk)
        return finder.find_top_k_paths()
    
    # OLMoE 特化策略：
    # 每次选出一条最优路径后，直接禁用该路径经过的所有专家，
    # 确保后续路径不会重复选择这些专家，从而鼓励覆盖更多不同的专家。
    selected: List[PathEntry] = []
    for _ in range(topk):
        finder = TopKPathFinder(graph, 1)
        best_path = finder.find_top_k_paths()
        if not best_path:
            break
        entry = best_path[0]
        selected.append(entry)

        # 直接禁用本路径上的所有专家，避免后续路径重复选择
        if entry.path_experts:
            graph.disable_experts(entry.path_experts)

    return selected


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def discover_sample_files(input_dir: Path, limit: int) -> List[Path]:
    pattern = re.compile(r"expert_analysis_results_sample_(\d+)\.json$")
    matches: List[Tuple[int, Path]] = []
    for name in os.listdir(input_dir):
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            matches.append((idx, input_dir / name))
    matches.sort(key=lambda x: x[0])
    if limit is not None and limit > 0:
        matches = matches[:limit]
    return [path for _, path in matches]


def preprocess_analysis_results(
    analysis_results: Dict,
    inspector: Optional[LayerOutlierInspector],
) -> Tuple[Dict, Set[Tuple[int, int]], Set[Tuple[int, int]], Dict[int, LayerOutlierResult]]:
    """Preprocess analysis results with optional outlier detection."""
    import copy

    filtered = copy.deepcopy(analysis_results)
    force_keep: Set[Tuple[int, int]] = set()
    force_remove: Set[Tuple[int, int]] = set()
    per_layer_details: Dict[int, LayerOutlierResult] = {}

    if inspector is None:
        return filtered, force_keep, force_remove, per_layer_details

    for key, layer_data in filtered.items():
        if not key.startswith("layer_"):
            continue
        layer_idx = int(key.split("_")[1])
        result = inspector.analyze_layer(layer_idx, layer_data)
        per_layer_details[layer_idx] = result

        for expert_idx in result.force_keep:
            force_keep.add((layer_idx, expert_idx))
        for expert_idx in result.force_remove:
            force_remove.add((layer_idx, expert_idx))

    force_remove -= force_keep

    return filtered, force_keep, force_remove, per_layer_details


def format_outlier_summary(per_layer_details: Dict[int, LayerOutlierResult]) -> str:
    lines = []
    for layer_idx in sorted(per_layer_details.keys()):
        details = per_layer_details[layer_idx]
        keep_str = (
            ", ".join(
                f"E{idx}({'; '.join(details.reasons[idx])})" for idx in sorted(details.force_keep)
            )
            or "None"
        )
        remove_str = (
            ", ".join(
                f"E{idx}({'; '.join(details.reasons[idx])})" for idx in sorted(details.force_remove)
            )
            or "None"
        )
        lines.append(f"    layer_{layer_idx}: keep[{keep_str}] remove[{remove_str}]")
    return "\n".join(lines)


def run_single_sample(
    analysis_results: Dict,
    topk: int,
    tau: float,
    inspector: Optional[LayerOutlierInspector],
    unique_paths: bool = False,
):
    """Run path search for a single sample."""
    filtered, force_keep, force_remove, details = preprocess_analysis_results(analysis_results, inspector)

    initial_disabled = force_keep | force_remove
    graph = MoEGraph(filtered, tau=tau, initial_disabled=initial_disabled.copy())

    top_k_paths = select_top_paths(graph, topk, unique_paths)

    return {
        "graph": graph,
        "top_k_paths": top_k_paths,
        "force_keep": force_keep,
        "force_remove": force_remove,
        "outlier_details": details,
    }


def build_masks_from_keep_set(
    keep_set: Set[Tuple[int, int]],
    num_layers: int,
    num_experts: int,
) -> Dict[str, List[int]]:
    masks: Dict[str, List[int]] = {}
    for layer_idx in range(num_layers):
        mask = [0] * num_experts
        for expert_idx in range(num_experts):
            if (layer_idx, expert_idx) in keep_set:
                mask[expert_idx] = 1
        masks[f"layer_{layer_idx}"] = mask
    return masks


def collect_kept_experts_from_paths(paths: List[PathEntry]) -> Set[Tuple[int, int]]:
    """Return the union of experts included in provided paths."""
    kept: Set[Tuple[int, int]] = set()
    for entry in paths:
        kept.update(entry.path_experts)
    return kept


def union_keep_for_k(
    per_sample_paths: List[List[PathEntry]],
    global_force_keep: Set[Tuple[int, int]],
    k: int,
) -> Set[Tuple[int, int]]:
    """Return keep set when taking first-k paths from each sample."""
    union: Set[Tuple[int, int]] = set(global_force_keep)
    for sample_paths in per_sample_paths:
        take = min(k, len(sample_paths))
        for entry in sample_paths[:take]:
            union.update(entry.path_experts)
    return union


def fill_masks_to_target(
    masks: Dict[str, List[int]],
    target_keep: int,
    force_remove: Set[Tuple[int, int]],
    num_layers: int,
    num_experts: int,
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Fill pruning masks to reach target_keep by flipping 0->1 (respecting force-remove).
    Returns (new_total_keep, added_experts).
    """
    current_keep = sum(sum(mask) for mask in masks.values())
    if current_keep >= target_keep:
        return current_keep, []

    deficit = target_keep - current_keep
    added: List[Tuple[int, int]] = []
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        mask = masks[layer_key]
        for expert_idx in range(num_experts):
            if mask[expert_idx] == 1:
                continue
            if (layer_idx, expert_idx) in force_remove:
                continue
            mask[expert_idx] = 1
            added.append((layer_idx, expert_idx))
            deficit -= 1
            if deficit == 0:
                total_keep = current_keep + len(added)
                return total_keep, added

    total_keep = current_keep + len(added)
    return total_keep, added


def run_path_search(input_dir: Path, config: PathSearchConfig) -> PathSearchResult:
    """Run frequency-based path search."""
    logger.info(f"Starting path search with config: {config}")

    sample_files = discover_sample_files(input_dir, config.limit)
    if not sample_files:
        raise RuntimeError(f"No expert_analysis_results_sample_*.json found in {input_dir}")

    logger.info(f"Discovered {len(sample_files)} sample files (limit={config.limit})")
    algo_label = "freq_rank_unique" if config.unique_paths else "freq_rank"
    logger.info(
        f"Algorithm: {algo_label}, target_keep={config.target_keep}, topk={config.topk}, tau={config.tau}"
    )
    if config.unique_paths:
        logger.info("Unique path selection enabled (v2, experts will not repeat across paths).")
        if getattr(config, "unique_path_mode", None):
            logger.info(f"Unique-path mode: {config.unique_path_mode}")

    inspector: Optional[LayerOutlierInspector] = None
    if config.enable_outlier_detection:
        inspector = LayerOutlierInspector(
            OutlierConfig(
                loss_iqr_multiplier=config.loss_iqr_multiplier,
                norm_iqr_multiplier=config.norm_iqr_multiplier,
                min_values=config.min_values,
            )
        )
    else:
        logger.info("Outlier detection disabled; skipping auto keep/remove heuristics.")

    layer_map_path = input_dir / "moe_layer_map.json"
    moe_layer_indices = None
    total_model_layers = None
    if layer_map_path.exists():
        try:
            with open(layer_map_path, "r", encoding="utf-8") as f:
                layer_map = json.load(f)
            moe_layer_indices = layer_map.get("moe_layer_indices")
            total_model_layers = layer_map.get("total_layers")
            logger.info(
                f"Loaded MoE layer map: {len(moe_layer_indices)} routed layers "
                f"out of {total_model_layers} total transformer layers."
            )
        except Exception as exc:
            logger.warning(f"Failed to load MoE layer map ({layer_map_path}): {exc}")
            moe_layer_indices = None
            total_model_layers = None
    else:
        logger.info("MoE layer map not found; assuming every transformer layer is MoE.")

    per_sample_paths: List[List[PathEntry]] = []
    per_sample_force_keep: List[Set[Tuple[int, int]]] = []
    per_sample_force_remove: List[Set[Tuple[int, int]]] = []
    per_sample_summary: List[Dict] = []
    representative_num_layers = None
    representative_num_experts = None

    for rank, path in enumerate(sample_files):
        logger.info(f"[{rank + 1}/{len(sample_files)}] Processing: {path}")
        analysis_results = load_json(str(path))
        result = run_single_sample(
            analysis_results,
            config.topk,
            config.tau,
            inspector,
            unique_paths=config.unique_paths,
        )

        graph = result["graph"]
        per_sample_paths.append(result["top_k_paths"])
        per_sample_force_keep.append(result["force_keep"])
        per_sample_force_remove.append(result["force_remove"])
        if representative_num_layers is None:
            representative_num_layers = graph.num_layers
            representative_num_experts = graph.num_experts

        per_sample_all_keep = result["force_keep"].copy()
        # 统计路径经过的专家的并集
        path_experts_union = collect_kept_experts_from_paths(result["top_k_paths"])
        per_sample_all_keep.update(path_experts_union)
        # 路径专家并集 + force_keep 的并集
        paths_plus_force_keep = path_experts_union | result["force_keep"]

        summary = {
            "file": path.name,
            "num_paths": len(result["top_k_paths"]),
            "num_experts_kept": len(per_sample_all_keep),
            "num_experts_in_paths_union": len(paths_plus_force_keep),  # 路径经过的专家 + force_keep 的并集数量
            "auto_keep": len(result["force_keep"]),
            "auto_remove": len(result["force_remove"]),
        }
        per_sample_summary.append(summary)

        logger.info(f"  Auto-keep experts: {summary['auto_keep']}")
        logger.info(f"  Auto-remove experts: {summary['auto_remove']}")
        logger.info(f"  Experts in paths union: {len(path_experts_union)}")
        logger.info(f"  Paths union + force_keep: {summary['num_experts_in_paths_union']}")
        logger.info(f"  Kept experts (force_keep + topk paths union): {len(per_sample_all_keep)}")
        if config.enable_outlier_detection:
            logger.info("  Outlier summary:")
            logger.info(format_outlier_summary(result["outlier_details"]))

    if representative_num_layers is None or representative_num_experts is None:
        raise RuntimeError("No representative model structure detected.")

    global_force_keep: Set[Tuple[int, int]] = set()
    for keep in per_sample_force_keep:
        global_force_keep.update(keep)

    global_force_remove: Set[Tuple[int, int]] = set()
    for remove in per_sample_force_remove:
        global_force_remove.update(remove)

    num_total_experts = representative_num_layers * representative_num_experts
    version_label: str
    algo_label: str
    branch_fields: Dict[str, Any] = {}
    zero_freq_kept: List[Tuple[int, int]] = []
    visited_experts_count = 0
    final_keep_set: Set[Tuple[int, int]]
    merged_masks: Dict[str, List[int]]
    total_keep = 0

    # 当 unique_paths=True 时，支持两种模式：
    # - adaptive_unique=True: 自适应 unique-path 聚合（当前 DeepSeek/OLMoE 默认）
    # - adaptive_unique=False: Mixtral 样式的频次排序，但路径仍然唯一
    if config.unique_paths and getattr(config, "adaptive_unique", True):
        version_label = "v7_unique_adaptive"
        algo_label = "freq_rank_unique_adaptive"
        logger.info("Running adaptive unique-path aggregation (path_searchv3 style).")

        keep_stats: List[Tuple[int, int]] = []
        union_cache: Dict[int, Set[Tuple[int, int]]] = {}
        selected_k = 0
        candidate_k = 0
        minimal_exceeds_case = False

        if per_sample_paths:
            max_available_k = max(len(paths) for paths in per_sample_paths)
            candidate_k = min(config.topk, max_available_k)
            if candidate_k == 0:
                logger.warning(
                    "All samples produced zero valid paths; falling back to force-keep experts only."
                )
            else:
                for k in range(1, candidate_k + 1):
                    union_set = union_keep_for_k(per_sample_paths, global_force_keep, k)
                    union_cache[k] = union_set
                    keep_stats.append((k, len(union_set)))

                logger.info("\nAdaptive unique-path sweep (k vs keep_count):")
                for k, count in keep_stats:
                    logger.info(f"  k={k}: keep_count={count}")

                minimal_exceeds_case = keep_stats[0][1] > config.target_keep if keep_stats else False
                if not minimal_exceeds_case:
                    for k, count in keep_stats:
                        if count <= config.target_keep:
                            selected_k = k

                if selected_k == 0:
                    if minimal_exceeds_case and keep_stats:
                        logger.error(
                            f"target_keep={config.target_keep} is smaller than keep count with k=1 "
                            f"({keep_stats[0][1]}). Falling back to k=1 without adjustments."
                        )
                        selected_k = 1
                    else:
                        selected_k = candidate_k
        else:
            logger.warning("No per-sample paths collected; using force-keep experts only.")

        if selected_k == 0:
            selected_keep_set = set(global_force_keep)
        else:
            selected_keep_set = set(union_cache.get(selected_k, global_force_keep))

        visited_experts_count = len(selected_keep_set)
        final_keep_set = set(selected_keep_set)

        fill_added: List[Tuple[int, int]] = []
        # 先构建初始 masks
        merged_masks = build_masks_from_keep_set(
            final_keep_set, representative_num_layers, representative_num_experts
        )
        total_keep = sum(sum(mask) for mask in merged_masks.values())
        if not minimal_exceeds_case and total_keep < config.target_keep:
            total_keep, fill_added = fill_masks_to_target(
                merged_masks,
                config.target_keep,
                global_force_remove,
                representative_num_layers,
                representative_num_experts,
            )
            if fill_added:
                logger.info(
                    f"Filled {len(fill_added)} additional experts to reach target_keep={config.target_keep}."
                )
                final_keep_set.update(fill_added)
                # 重新构建 masks，包含 fill_added 的专家
                merged_masks = build_masks_from_keep_set(
                    final_keep_set, representative_num_layers, representative_num_experts
                )
                total_keep = sum(sum(mask) for mask in merged_masks.values())
        elif minimal_exceeds_case and total_keep > config.target_keep:
            logger.info(
                "Selected k=1 already exceeds target_keep; skip filling per adaptive unique-path rule."
            )

        zero_freq_kept = sorted(fill_added)
        branch_fields = {
            "selected_k": selected_k,
            "max_k": candidate_k,
            "keep_stats": [{"k": k, "keep_count": count} for k, count in keep_stats],
            "minimal_exceeds_case": minimal_exceeds_case,
            "fill_added_experts": zero_freq_kept,
        }
    else:
        # 频次排序模式（固定 topk）。对于 DeepSeek/OLMoE + unique_paths=True，
        # 这里依然使用不重复路径（由 select_top_paths 控制），只是聚合方式改为频次统计。
        version_label = (
            "v6_freq_rank_unique_fixed" if config.unique_paths else "v6_freq_rank"
        )
        algo_label = "freq_rank_unique_fixed" if config.unique_paths else "freq_rank"

        expert_freq: Dict[Tuple[int, int], int] = defaultdict(int)
        for sample_paths in per_sample_paths:
            for entry in sample_paths:
                for expert in entry.path_experts:
                    expert_freq[expert] += 1

        logger.info("\nFrequency statistics (top few experts):")
        freq_items = sorted(expert_freq.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        for (layer_idx, expert_idx), cnt in freq_items[:20]:
            logger.info(f"  layer_{layer_idx}, expert_{expert_idx}: freq={cnt}")

        visited_experts = set(expert_freq.keys())
        paths_plus_force_keep_global = visited_experts | global_force_keep

        logger.info("")
        logger.info("=" * 80)
        logger.info("Path Search Expert Count Comparison")
        logger.info("=" * 80)
        logger.info(f"Total experts visited by paths (paths union): {len(visited_experts)}")
        logger.info(f"Global force-keep experts: {len(global_force_keep)}")
        logger.info(f"Path experts + force-keep experts total: {len(paths_plus_force_keep_global)}")
        logger.info(f"Target keep experts (target_keep): {config.target_keep}")
        logger.info("")

        if len(paths_plus_force_keep_global) >= config.target_keep:
            logger.info(
                f"✓ Path experts total ({len(paths_plus_force_keep_global)}) >= target keep ({config.target_keep})"
            )
            logger.info("  Sufficient experts from paths to reach target")
        else:
            logger.warning(
                f"⚠ Path experts total ({len(paths_plus_force_keep_global)}) < target keep ({config.target_keep})"
            )
            logger.warning(
                f"  Insufficient path experts, will keep all path experts ({len(paths_plus_force_keep_global)} total)"
            )
        logger.info("=" * 80)
        logger.info("")

        base_keep: Set[Tuple[int, int]] = set(global_force_keep)
        logger.info(f"\nGlobal force-keep experts: {len(base_keep)} / {num_total_experts}")

        if len(base_keep) >= config.target_keep:
            logger.warning(
                f"Number of force-keep experts ({len(base_keep)}) >= target_keep ({config.target_keep}). "
                "Will keep all force-keep experts and prune all others."
            )
            final_keep_set = base_keep
        else:
            remaining_slots = config.target_keep - len(base_keep)
            logger.info(f"Need to select {remaining_slots} more experts based on frequency ranking.")

            candidates: List[Tuple[Tuple[int, int], int]] = []
            for layer_idx in range(representative_num_layers):
                for expert_idx in range(representative_num_experts):
                    key = (layer_idx, expert_idx)
                    if key in base_keep:
                        continue
                    cnt = expert_freq.get(key, 0)
                    candidates.append((key, cnt))

            candidates.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))

            selected_from_rank: List[Tuple[int, int]] = []
            for key, cnt in candidates:
                selected_from_rank.append(key)
                if len(selected_from_rank) >= remaining_slots:
                    break

            if len(selected_from_rank) < remaining_slots:
                logger.warning(
                    f"Not enough candidates to reach target_keep={config.target_keep}. "
                    f"Only {len(base_keep) + len(selected_from_rank)} experts available."
                )

            final_keep_set = base_keep.union(selected_from_rank)

        merged_masks = build_masks_from_keep_set(
            final_keep_set, representative_num_layers, representative_num_experts
        )
        total_keep = sum(sum(mask) for mask in merged_masks.values())
        visited_experts_count = len(visited_experts)
        
        # 验证专家保留逻辑：确保 final_keep_set 中的所有专家都在 masks 中被标记为保留
        verification_passed = True
        missing_experts = []
        for layer_idx in range(representative_num_layers):
            layer_key = f"layer_{layer_idx}"
            mask = merged_masks[layer_key]
            for expert_idx in range(representative_num_experts):
                if (layer_idx, expert_idx) in final_keep_set:
                    if mask[expert_idx] != 1:
                        verification_passed = False
                        missing_experts.append((layer_idx, expert_idx))
        
        if not verification_passed:
            logger.error(
                f"Expert retention verification failed! Found {len(missing_experts)} experts that should be kept but are not marked in masks:"
            )
            for layer_idx, expert_idx in missing_experts[:10]:  # Show only first 10
                logger.error(f"  layer_{layer_idx}, expert_{expert_idx}")
            if len(missing_experts) > 10:
                logger.error(f"  ... and {len(missing_experts) - 10} more experts")
            raise RuntimeError("Expert retention verification failed! Please check build_masks_from_keep_set function.")
        else:
            logger.info(f"✓ Expert retention verification passed: all {len(final_keep_set)} selected experts are correctly marked as kept in masks")
        zero_freq_kept = [
            (layer_idx, expert_idx)
            for layer_idx, expert_idx in sorted(final_keep_set)
            if expert_freq.get((layer_idx, expert_idx), 0) == 0
        ]

    experts_to_prune: Dict[str, List[int]] = {}
    for layer_idx in range(representative_num_layers):
        layer_key = f"layer_{layer_idx}"
        mask = merged_masks[layer_key]
        prune_indices = [idx for idx, bit in enumerate(mask) if bit == 0]
        experts_to_prune[layer_key] = prune_indices

    # 标记不同 unique_paths 策略，方便在文件名/JSON 中区分 DeepSeek 的两种模式
    mode_suffix = ""
    unique_mode_value = None
    if config.unique_paths:
        unique_mode_value = config.unique_path_mode or ("adaptive" if getattr(config, "adaptive_unique", True) else "freq_topk")
        mode_suffix = "_adaptive" if unique_mode_value == "adaptive" else "_freq_topk"

    merged_output = {
        "version": version_label,
        "algo": algo_label,
        "topk": config.topk,
        "tau": config.tau,
        "target_keep": config.target_keep,
        "samples_processed": len(sample_files),
        "total_experts": num_total_experts,
        "experts_to_keep": total_keep,
        "experts_to_prune": num_total_experts - total_keep,
        "pruning_ratio": 1.0 - float(total_keep) / float(num_total_experts),
        "pruning_masks": merged_masks,
        "experts_to_prune_indices": experts_to_prune,
        "per_sample_summary": per_sample_summary,
        "global_force_keep": sorted(list(global_force_keep)),
        "global_force_remove": sorted(list(global_force_remove)),
        "visited_experts_count": visited_experts_count,
        "zero_freq_kept_experts": zero_freq_kept,
        "outlier_config": {
            "loss_iqr_multiplier": config.loss_iqr_multiplier,
            "norm_iqr_multiplier": config.norm_iqr_multiplier,
            "min_values": config.min_values,
        },
        "unique_paths": config.unique_paths,
        "unique_path_mode": unique_mode_value if config.unique_paths else None,
        "moe_layer_indices": moe_layer_indices
        if moe_layer_indices is not None
        else list(range(representative_num_layers)),
        "total_model_layers": total_model_layers
        if total_model_layers is not None
        else representative_num_layers,
    }
    merged_output.update(branch_fields)

    if config.unique_paths:
        output_suffix = f"freq_rank_unique{mode_suffix}"
    else:
        output_suffix = "freq_rank"
    
    # 检查是否是合并模式（通过检查输入目录名或样本文件数量）
    # 如果只有一个样本文件，可能是合并模式，但为了更准确，我们检查目录名
    is_merged_mode = "_merged" in str(input_dir)
    merged_suffix = "_merged" if is_merged_mode else ""

    output_path = input_dir / f"expert_pruning_{output_suffix}_target{config.target_keep}_topk{config.topk}_{len(sample_files)}{merged_suffix}.json"

    logger.info(f"Saving merged pruning results to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(merged_output, f, indent=2)

    logger.info("Path search complete!")
    return PathSearchResult(mask_path=output_path)

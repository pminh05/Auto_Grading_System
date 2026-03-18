"""Graph comparator: compare two CFGs and report structural differences."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import networkx as nx

from app.models.schemas import ComparisonResult


class GraphComparator:
    """Compare a reference CFG against a student CFG and quantify differences.

    The comparison is based on node-type multisets and edge-type pairs rather
    than exact node IDs (which differ between independently built graphs).
    This gives a *semantic* similarity measure that is robust to trivial
    reorderings.
    """

    def compare(self, ref_graph: nx.DiGraph, student_graph: nx.DiGraph) -> ComparisonResult:
        """Compare *ref_graph* against *student_graph* and return a result.

        Args:
            ref_graph: CFG built from the reference solution.
            student_graph: CFG built from the student code.

        Returns:
            A :class:`~app.models.schemas.ComparisonResult` describing
            missing/extra nodes and edges, and an overall similarity score.
        """
        ref_nodes = self._node_type_list(ref_graph)
        stu_nodes = self._node_type_list(student_graph)

        missing_nodes = self._list_difference(ref_nodes, stu_nodes)
        extra_nodes = self._list_difference(stu_nodes, ref_nodes)

        ref_edges = self._edge_type_list(ref_graph)
        stu_edges = self._edge_type_list(student_graph)

        missing_edges = self._list_difference(ref_edges, stu_edges)
        extra_edges = self._list_difference(stu_edges, ref_edges)

        similarity = self._similarity(ref_nodes, stu_nodes, ref_edges, stu_edges)

        # Build rich diff details with full labels
        node_diff_detail = self._build_node_diff_detail(ref_graph, student_graph)
        edge_diff_detail = self._build_edge_diff_detail(missing_edges, extra_edges)

        return ComparisonResult(
            missing_nodes=[{"type": n} for n in missing_nodes],
            extra_nodes=[{"type": n} for n in extra_nodes],
            missing_edges=[{"type": e} for e in missing_edges],
            extra_edges=[{"type": e} for e in extra_edges],
            similarity_score=round(similarity, 4),
            node_diff_detail=node_diff_detail,
            edge_diff_detail=edge_diff_detail,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_type_list(graph: nx.DiGraph) -> List[str]:
        """Return a sorted list of node *type* attributes from *graph*.

        Ignores ENTRY and EXIT sentinel nodes used during graph construction.
        """
        return sorted(
            attrs.get("type", "")
            for _, attrs in graph.nodes(data=True)
            if attrs.get("type") not in ("ENTRY", "EXIT")
        )

    @staticmethod
    def _edge_type_list(graph: nx.DiGraph) -> List[str]:
        """Return a sorted list of ``(src_type, dst_type)`` strings from *graph*."""
        result: List[str] = []
        for src, dst in graph.edges():
            src_type = graph.nodes[src].get("type", "")
            dst_type = graph.nodes[dst].get("type", "")
            if src_type in ("ENTRY", "EXIT") or dst_type in ("ENTRY", "EXIT"):
                continue
            result.append(f"{src_type}->{dst_type}")
        return sorted(result)

    @staticmethod
    def _list_difference(a: List[str], b: List[str]) -> List[str]:
        """Return elements in *a* that are not in *b* (multiset difference)."""
        b_copy = list(b)
        diff: List[str] = []
        for item in a:
            if item in b_copy:
                b_copy.remove(item)
            else:
                diff.append(item)
        return diff

    def _similarity(
        self,
        ref_nodes: List[str],
        stu_nodes: List[str],
        ref_edges: List[str],
        stu_edges: List[str],
    ) -> float:
        """Compute a Jaccard-like similarity score.

        The score is the weighted average of node similarity and edge
        similarity, both weighted equally.

        Returns:
            Float in [0.0, 1.0].
        """
        node_sim = self._jaccard(ref_nodes, stu_nodes)
        edge_sim = self._jaccard(ref_edges, stu_edges)
        if not ref_nodes and not ref_edges:
            return 1.0
        return (node_sim + edge_sim) / 2.0

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        """Multiset Jaccard similarity sử dụng Counter.

        Dùng Counter thay vì set để giữ nguyên thông tin số lần xuất hiện
        của mỗi loại node/edge, tránh trường hợp similarity luôn ~1.0.
        """
        if not a and not b:
            return 1.0
        counter_a = Counter(a)
        counter_b = Counter(b)
        # Intersection: min of counts
        intersection = sum((counter_a & counter_b).values())
        # Union: max of counts
        union = sum((counter_a | counter_b).values())
        if union == 0:
            return 1.0
        return intersection / union

    def _build_node_diff_detail(
        self,
        ref_graph: nx.DiGraph,
        student_graph: nx.DiGraph,
    ) -> List[Dict[str, Any]]:
        """Build a rich diff list for nodes, including full labels.

        Returns a list of dicts with keys: ``direction`` (``"missing"`` or
        ``"extra"``), ``type``, and ``label``.
        """
        result: List[Dict[str, Any]] = []

        ref_node_details = [
            {"type": attrs.get("type", ""), "label": attrs.get("label", "")}
            for _, attrs in ref_graph.nodes(data=True)
            if attrs.get("type") not in ("ENTRY", "EXIT")
        ]
        stu_node_details = [
            {"type": attrs.get("type", ""), "label": attrs.get("label", "")}
            for _, attrs in student_graph.nodes(data=True)
            if attrs.get("type") not in ("ENTRY", "EXIT")
        ]

        # Use Counter for O(n) type-level matching
        ref_type_counts = Counter(n["type"] for n in ref_node_details)
        stu_type_counts = Counter(n["type"] for n in stu_node_details)

        # Types present more in ref than in student → missing
        missing_counts = ref_type_counts - stu_type_counts
        # Types present more in student than in ref → extra
        extra_counts = stu_type_counts - ref_type_counts

        # Collect one representative detail per missing type occurrence
        missing_remaining = dict(missing_counts)
        for node in ref_node_details:
            t = node["type"]
            if missing_remaining.get(t, 0) > 0:
                result.append({"direction": "missing", **node})
                missing_remaining[t] -= 1

        # Collect one representative detail per extra type occurrence
        extra_remaining = dict(extra_counts)
        for node in stu_node_details:
            t = node["type"]
            if extra_remaining.get(t, 0) > 0:
                result.append({"direction": "extra", **node})
                extra_remaining[t] -= 1

        return result

    @staticmethod
    def _build_edge_diff_detail(
        missing_edges: List[str],
        extra_edges: List[str],
    ) -> List[Dict[str, Any]]:
        """Build a rich diff list for edges.

        Returns a list of dicts with keys: ``direction`` and ``type``.
        """
        result: List[Dict[str, Any]] = []
        for edge_type in missing_edges:
            result.append({"direction": "missing", "type": edge_type})
        for edge_type in extra_edges:
            result.append({"direction": "extra", "type": edge_type})
        return result

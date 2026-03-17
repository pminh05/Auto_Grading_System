"""Graph comparator: compare two CFGs and report structural differences."""

from __future__ import annotations

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

        return ComparisonResult(
            missing_nodes=[{"type": n} for n in missing_nodes],
            extra_nodes=[{"type": n} for n in extra_nodes],
            missing_edges=[{"type": e} for e in missing_edges],
            extra_edges=[{"type": e} for e in extra_edges],
            similarity_score=round(similarity, 4),
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
        """Multiset Jaccard similarity between two lists."""
        if not a and not b:
            return 1.0
        set_a, set_b = set(a), set(b)
        intersection = set_a & set_b
        union = set_a | set_b
        if not union:
            return 1.0
        return len(intersection) / len(union)

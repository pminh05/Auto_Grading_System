"""Graph comparator: compare two PDGs and report structural differences."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import networkx as nx

from app.models.schemas import ComparisonResult


class GraphComparator:
    """Compare a reference PDG against a student PDG and quantify differences.

    The comparison is based on node-type multisets and edge-type triples
    (``src_type-edge_type->dst_type``) rather than exact node IDs, giving a
    *semantic* similarity measure robust to trivial reorderings.

    Key fixes vs. the previous CFG-only comparator:
    - Uses **multiset** Jaccard (``Counter``-based) so that differing *counts*
      of the same node/edge type correctly reduce the similarity score.
    - Returns ``0.0`` (not ``1.0``) when both graphs are empty, as there is no
      structural evidence of a correct solution.
    - Includes the edge ``type`` attribute (``FLOW``, ``DATA``,
      ``BRANCH_TRUE``, ``BRANCH_FALSE``, ``LOOP_BACK``) in the edge-type key
      for richer structural comparison.
    """

    def compare(self, ref_graph: nx.DiGraph, student_graph: nx.DiGraph) -> ComparisonResult:
        """Compare *ref_graph* against *student_graph* and return a result.

        Args:
            ref_graph: PDG built from the reference solution.
            student_graph: PDG built from the student code.

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
        """Return a sorted list of ``src_type-edge_type->dst_type`` strings.

        Edges involving ENTRY or EXIT sentinel nodes are excluded.  The edge
        ``type`` attribute (e.g. ``FLOW``, ``DATA``, ``BRANCH_TRUE``) is
        included to capture richer structural information from the PDG.
        """
        result: List[str] = []
        for src, dst, attrs in graph.edges(data=True):
            src_type = graph.nodes[src].get("type", "")
            dst_type = graph.nodes[dst].get("type", "")
            if src_type in ("ENTRY", "EXIT") or dst_type in ("ENTRY", "EXIT"):
                continue
            edge_type = attrs.get("type", "FLOW")
            result.append(f"{src_type}-{edge_type}->{dst_type}")
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
        """Compute a multiset-Jaccard similarity score.

        The score is the weighted average of node similarity and edge
        similarity, both weighted equally.

        Returns ``0.0`` when both graphs have no meaningful content (empty
        node and edge lists), rather than the incorrect ``1.0`` that a naïve
        implementation would return.

        Returns:
            Float in [0.0, 1.0].
        """
        # If the reference has no meaningful content, return 0 — we cannot
        # assess whether the student code is correct against an empty reference.
        if not ref_nodes and not ref_edges:
            return 0.0

        node_sim = self._jaccard(ref_nodes, stu_nodes)
        edge_sim = self._jaccard(ref_edges, stu_edges)
        return (node_sim + edge_sim) / 2.0

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        """Multiset Jaccard similarity between two lists.

        Uses ``Counter``-based intersection (``min`` counts) and union
        (``max`` counts) so that differing multiplicities lower the score.
        Returns ``0.0`` when both lists are empty (no evidence of similarity).
        """
        count_a = Counter(a)
        count_b = Counter(b)
        all_keys = set(count_a) | set(count_b)
        intersection = sum(min(count_a[k], count_b[k]) for k in all_keys)
        union = sum(max(count_a[k], count_b[k]) for k in all_keys)
        if union == 0:
            return 0.0
        return intersection / union

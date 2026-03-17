"""Error localization and severity classification based on CFG comparison results."""

from __future__ import annotations

from typing import List

from app.models.schemas import ClassifiedError, ComparisonResult, ErrorSeverity, ErrorType


class ErrorClassifier:
    """Classify graph differences into typed, severity-rated errors.

    Rules are deterministic and based on the node/edge types reported by
    :class:`~app.graph.comparator.GraphComparator`.
    """

    # Node types that map to specific error categories
    _LOOP_TYPES = {"FOR", "WHILE"}
    _SYNTAX_TYPES = {"FUNCDEF", "IMPORT", "ANN_ASSIGN"}
    _DATA_TYPES = {"ASSIGN", "AUG_ASSIGN"}

    def classify(self, comparison_result: ComparisonResult) -> List[ClassifiedError]:
        """Produce a list of classified errors from *comparison_result*.

        Args:
            comparison_result: Output of
                :meth:`~app.graph.comparator.GraphComparator.compare`.

        Returns:
            A list of :class:`~app.models.schemas.ClassifiedError` objects,
            ordered from most to least severe.
        """
        errors: List[ClassifiedError] = []

        # ---- Missing nodes (things the student forgot to implement) ----
        for node_info in comparison_result.missing_nodes:
            node_type = node_info.get("type", "UNKNOWN")
            errors.append(self._classify_missing_node(node_type))

        # ---- Extra nodes (things the student added unnecessarily) ------
        for node_info in comparison_result.extra_nodes:
            node_type = node_info.get("type", "UNKNOWN")
            errors.append(self._classify_extra_node(node_type))

        # ---- Missing edges (wrong / absent control flow) ---------------
        for edge_info in comparison_result.missing_edges:
            edge_type = edge_info.get("type", "->")
            errors.append(self._classify_missing_edge(edge_type))

        # ---- Extra edges (spurious control flow paths) -----------------
        for edge_info in comparison_result.extra_edges:
            edge_type = edge_info.get("type", "->")
            errors.append(self._classify_extra_edge(edge_type))

        # Sort: CRITICAL first, then MAJOR, then MINOR
        severity_order = {ErrorSeverity.CRITICAL: 0, ErrorSeverity.MAJOR: 1, ErrorSeverity.MINOR: 2}
        errors.sort(key=lambda e: severity_order[e.severity])
        return errors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_missing_node(self, node_type: str) -> ClassifiedError:
        """Classify a missing node."""
        if node_type == "IF":
            return ClassifiedError(
                error_type=ErrorType.ALGORITHM,
                severity=ErrorSeverity.MAJOR,
                description=f"Missing conditional branch (IF statement) present in reference solution.",
                node_label=node_type,
            )
        if node_type in self._LOOP_TYPES:
            return ClassifiedError(
                error_type=ErrorType.LOOP,
                severity=ErrorSeverity.MAJOR,
                description=f"Missing loop construct ({node_type}) present in reference solution.",
                node_label=node_type,
            )
        if node_type == "RETURN":
            return ClassifiedError(
                error_type=ErrorType.ALGORITHM,
                severity=ErrorSeverity.CRITICAL,
                description="Missing RETURN statement — function may not return the correct value.",
                node_label=node_type,
            )
        if node_type == "FUNCDEF":
            return ClassifiedError(
                error_type=ErrorType.SYNTAX,
                severity=ErrorSeverity.CRITICAL,
                description="Missing function definition that exists in the reference solution.",
                node_label=node_type,
            )
        if node_type in self._DATA_TYPES:
            return ClassifiedError(
                error_type=ErrorType.DATA_HANDLING,
                severity=ErrorSeverity.MINOR,
                description=f"Missing assignment operation ({node_type}) — possible data handling omission.",
                node_label=node_type,
            )
        return ClassifiedError(
            error_type=ErrorType.ALGORITHM,
            severity=ErrorSeverity.MINOR,
            description=f"Missing operation of type '{node_type}' found in reference solution.",
            node_label=node_type,
        )

    def _classify_extra_node(self, node_type: str) -> ClassifiedError:
        """Classify an extra (unexpected) node."""
        if node_type in self._LOOP_TYPES:
            return ClassifiedError(
                error_type=ErrorType.LOOP,
                severity=ErrorSeverity.MAJOR,
                description=f"Extra loop construct ({node_type}) not present in reference solution — possible infinite loop or unnecessary iteration.",
                node_label=node_type,
            )
        if node_type == "IF":
            return ClassifiedError(
                error_type=ErrorType.ALGORITHM,
                severity=ErrorSeverity.MINOR,
                description="Extra conditional branch not present in reference — may indicate unnecessary guard.",
                node_label=node_type,
            )
        return ClassifiedError(
            error_type=ErrorType.ALGORITHM,
            severity=ErrorSeverity.MINOR,
            description=f"Extra operation of type '{node_type}' not in reference solution.",
            node_label=node_type,
        )

    def _classify_missing_edge(self, edge_type: str) -> ClassifiedError:
        """Classify a missing control-flow edge."""
        src, _, dst = edge_type.partition("->")
        if "FOR" in src or "WHILE" in src or "FOR" in dst or "WHILE" in dst:
            return ClassifiedError(
                error_type=ErrorType.LOOP,
                severity=ErrorSeverity.MAJOR,
                description=f"Missing control-flow edge '{edge_type}' — loop structure may be incorrect.",
            )
        if "IF" in src or "IF" in dst:
            return ClassifiedError(
                error_type=ErrorType.ALGORITHM,
                severity=ErrorSeverity.MAJOR,
                description=f"Missing conditional control-flow edge '{edge_type}'.",
            )
        return ClassifiedError(
            error_type=ErrorType.ALGORITHM,
            severity=ErrorSeverity.MINOR,
            description=f"Missing control-flow edge '{edge_type}'.",
        )

    def _classify_extra_edge(self, edge_type: str) -> ClassifiedError:
        """Classify an extra (unexpected) control-flow edge."""
        return ClassifiedError(
            error_type=ErrorType.ALGORITHM,
            severity=ErrorSeverity.MINOR,
            description=f"Extra control-flow edge '{edge_type}' not present in reference solution.",
        )

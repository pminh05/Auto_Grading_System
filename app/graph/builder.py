"""Control Flow Graph (CFG) builder from Python source code using the ast module."""

from __future__ import annotations

import ast
from typing import Any, Dict, Optional

import networkx as nx


class GraphBuilder:
    """Builds a Control Flow Graph (CFG) from Python source code.

    Each node represents a single operation (assignment, loop header,
    conditional branch, function definition, return, or function call).
    Each directed edge represents a control-flow transition between operations.

    Attributes:
        _counter: Auto-incrementing node ID counter.
        _graph: The directed graph being constructed.
    """

    def __init__(self) -> None:
        """Initialise an empty builder."""
        self._counter: int = 0
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, source_code: str) -> nx.DiGraph:
        """Parse *source_code* and return a directed CFG.

        Args:
            source_code: Valid Python 3 source code.

        Returns:
            A :class:`networkx.DiGraph` whose nodes carry ``type``,
            ``lineno``, and ``label`` attributes.

        Raises:
            SyntaxError: If *source_code* cannot be parsed.
        """
        self._counter = 0
        self._graph = nx.DiGraph()
        tree = ast.parse(source_code)
        entry = self._add_node("ENTRY", 0, "ENTRY")
        last_nodes = [entry]
        last_nodes = self._visit_stmts(tree.body, last_nodes)
        exit_node = self._add_node("EXIT", 0, "EXIT")
        for node in last_nodes:
            self._graph.add_edge(node, exit_node)
        return self._graph

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_node(self, node_type: str, lineno: int, label: str) -> int:
        """Create a new graph node and return its ID.

        Args:
            node_type: Semantic type string (e.g. ``"ASSIGN"``, ``"IF"``).
            lineno: Source line number (0 if not applicable).
            label: Human-readable label for the node.

        Returns:
            The integer node ID.
        """
        node_id = self._counter
        self._counter += 1
        self._graph.add_node(node_id, type=node_type, lineno=lineno, label=label)
        return node_id

    def _visit_stmts(self, stmts: list, prev_nodes: list[int]) -> list[int]:
        """Walk a list of AST statements and wire them sequentially.

        Args:
            stmts: List of :class:`ast.stmt` nodes.
            prev_nodes: The graph nodes that should have edges *into* the
                first statement.

        Returns:
            The list of graph nodes that are "open" (no successor yet)
            after processing all statements.
        """
        current = prev_nodes
        for stmt in stmts:
            current = self._visit_stmt(stmt, current)
        return current

    def _visit_stmt(self, stmt: ast.stmt, prev_nodes: list[int]) -> list[int]:
        """Dispatch a single AST statement to the appropriate handler.

        Args:
            stmt: An AST statement node.
            prev_nodes: Predecessor graph nodes.

        Returns:
            Open successor graph nodes after this statement.
        """
        handler_map = {
            ast.Assign: self._handle_assign,
            ast.AugAssign: self._handle_aug_assign,
            ast.AnnAssign: self._handle_ann_assign,
            ast.If: self._handle_if,
            ast.For: self._handle_for,
            ast.While: self._handle_while,
            ast.FunctionDef: self._handle_funcdef,
            ast.AsyncFunctionDef: self._handle_funcdef,
            ast.Return: self._handle_return,
            ast.Expr: self._handle_expr,
            ast.Import: self._handle_import,
            ast.ImportFrom: self._handle_import,
        }
        for ast_type, handler in handler_map.items():
            if isinstance(stmt, ast_type):
                return handler(stmt, prev_nodes)
        # Generic fallback for unrecognised statements
        return self._handle_generic(stmt, prev_nodes)

    # --- Individual statement handlers ----------------------------------

    def _connect_from(self, prev_nodes: list[int], new_node: int) -> None:
        """Add edges from every predecessor to *new_node*."""
        for p in prev_nodes:
            self._graph.add_edge(p, new_node)

    def _handle_assign(self, stmt: ast.Assign, prev_nodes: list[int]) -> list[int]:
        targets = ", ".join(ast.unparse(t) for t in stmt.targets)
        label = f"ASSIGN: {targets} = {ast.unparse(stmt.value)}"
        node = self._add_node("ASSIGN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        return [node]

    def _handle_aug_assign(self, stmt: ast.AugAssign, prev_nodes: list[int]) -> list[int]:
        op = type(stmt.op).__name__.replace("Add", "+=").replace("Sub", "-=").replace("Mult", "*=").replace("Div", "/=")
        label = f"AUG_ASSIGN: {ast.unparse(stmt.target)} {op} {ast.unparse(stmt.value)}"
        node = self._add_node("AUG_ASSIGN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        return [node]

    def _handle_ann_assign(self, stmt: ast.AnnAssign, prev_nodes: list[int]) -> list[int]:
        label = f"ANN_ASSIGN: {ast.unparse(stmt.target)}: {ast.unparse(stmt.annotation)}"
        if stmt.value:
            label += f" = {ast.unparse(stmt.value)}"
        node = self._add_node("ANN_ASSIGN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        return [node]

    def _handle_if(self, stmt: ast.If, prev_nodes: list[int]) -> list[int]:
        cond_label = f"IF: {ast.unparse(stmt.test)}"
        cond_node = self._add_node("IF", stmt.lineno, cond_label)
        self._connect_from(prev_nodes, cond_node)

        # True branch
        true_exits = self._visit_stmts(stmt.body, [cond_node])

        # False / elif branch
        if stmt.orelse:
            false_exits = self._visit_stmts(stmt.orelse, [cond_node])
        else:
            false_exits = [cond_node]

        return true_exits + false_exits

    def _handle_for(self, stmt: ast.For, prev_nodes: list[int]) -> list[int]:
        label = f"FOR: {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}"
        loop_node = self._add_node("FOR", stmt.lineno, label)
        self._connect_from(prev_nodes, loop_node)

        body_exits = self._visit_stmts(stmt.body, [loop_node])
        # Back-edge from body to loop header
        for node in body_exits:
            self._graph.add_edge(node, loop_node)

        return [loop_node]  # exit edge from loop header

    def _handle_while(self, stmt: ast.While, prev_nodes: list[int]) -> list[int]:
        label = f"WHILE: {ast.unparse(stmt.test)}"
        loop_node = self._add_node("WHILE", stmt.lineno, label)
        self._connect_from(prev_nodes, loop_node)

        body_exits = self._visit_stmts(stmt.body, [loop_node])
        for node in body_exits:
            self._graph.add_edge(node, loop_node)

        return [loop_node]

    def _handle_funcdef(
        self,
        stmt: ast.FunctionDef | ast.AsyncFunctionDef,
        prev_nodes: list[int],
    ) -> list[int]:
        label = f"FUNCDEF: {stmt.name}({', '.join(a.arg for a in stmt.args.args)})"
        func_node = self._add_node("FUNCDEF", stmt.lineno, label)
        self._connect_from(prev_nodes, func_node)
        # Walk the function body connected from the func_def node
        self._visit_stmts(stmt.body, [func_node])
        return [func_node]

    def _handle_return(self, stmt: ast.Return, prev_nodes: list[int]) -> list[int]:
        value = ast.unparse(stmt.value) if stmt.value else "None"
        label = f"RETURN: {value}"
        node = self._add_node("RETURN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        return [node]

    def _handle_expr(self, stmt: ast.Expr, prev_nodes: list[int]) -> list[int]:
        """Handle expression statements (typically function calls)."""
        if isinstance(stmt.value, ast.Call):
            func_name = ast.unparse(stmt.value.func)
            label = f"CALL: {ast.unparse(stmt.value)}"
            node = self._add_node("CALL", stmt.lineno, label)
        else:
            label = f"EXPR: {ast.unparse(stmt.value)}"
            node = self._add_node("EXPR", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        return [node]

    def _handle_import(self, stmt: ast.stmt, prev_nodes: list[int]) -> list[int]:
        label = f"IMPORT: {ast.unparse(stmt)}"
        node = self._add_node("IMPORT", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        return [node]

    def _handle_generic(self, stmt: ast.stmt, prev_nodes: list[int]) -> list[int]:
        label = f"STMT: {ast.unparse(stmt)}"
        node = self._add_node("STMT", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        return [node]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def node_attrs(graph: nx.DiGraph, node_id: int) -> Dict[str, Any]:
        """Return attribute dictionary for a node, defaulting to empty strings.

        Args:
            graph: The CFG graph.
            node_id: Integer node identifier.

        Returns:
            Dictionary with at least ``type``, ``lineno``, and ``label`` keys.
        """
        return graph.nodes.get(node_id, {"type": "", "lineno": 0, "label": ""})

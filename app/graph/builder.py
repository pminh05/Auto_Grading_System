"""Program Dependence Graph (PDG) builder from Python source code using the ast module.

Combines Control Flow Graph (CFG) edges with Data Dependence Graph (DDG) edges to
produce a richer structural representation for code comparison.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, Set

import networkx as nx

# Mapping from AST operator class name to operator string for AugAssign
_AUG_OP_MAP: dict[str, str] = {
    "Add": "+=", "Sub": "-=", "Mult": "*=", "Div": "/=",
    "Mod": "%=", "Pow": "**=", "FloorDiv": "//=", "MatMult": "@=",
    "BitAnd": "&=", "BitOr": "|=", "BitXor": "^=",
    "LShift": "<<=", "RShift": ">>=",
}


class GraphBuilder:
    """Builds a Program Dependence Graph (PDG) from Python source code.

    The PDG combines:
    - **Control Flow edges** (CFG): sequential execution, branches, loops.
    - **Data Dependence edges** (DDG): variable ``x`` defined at node A and
      used at node B generates a DATA edge A → B.

    Edge ``type`` attribute values:
    - ``FLOW``         — sequential control flow
    - ``BRANCH_TRUE``  — true branch of an if/elif
    - ``BRANCH_FALSE`` — false branch of an if/elif (or fall-through)
    - ``LOOP_BACK``    — back-edge from loop body to loop header
    - ``DATA``         — data dependency (variable use-def chain)

    Node ``type`` attribute values include: ``ENTRY``, ``EXIT``,
    ``IMPORT``, ``ASSIGN``, ``AUG_ASSIGN``, ``ANN_ASSIGN``, ``IF``,
    ``FOR``, ``WHILE``, ``FUNCDEF``, ``RETURN``, ``CALL``, ``EXPR``,
    ``STMT``.

    Attributes:
        _counter: Auto-incrementing node ID counter.
        _graph: The directed graph being constructed.
        _var_defs: Maps variable name → node ID of its most recent definition.
        _pending_edge_types: Maps node ID → edge type to use for its *next*
            outgoing edge (used to inject BRANCH_TRUE/FALSE without threading
            the type through every recursive call).
    """

    def __init__(self) -> None:
        """Initialise an empty builder."""
        self._counter: int = 0
        self._graph: nx.DiGraph = nx.DiGraph()
        self._var_defs: dict[str, int] = {}
        self._pending_edge_types: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, source_code: str) -> nx.DiGraph:
        """Parse *source_code* and return a directed PDG.

        Args:
            source_code: Valid Python 3 source code.

        Returns:
            A :class:`networkx.DiGraph` whose nodes carry ``type``,
            ``lineno``, and ``label`` attributes, and whose edges carry
            a ``type`` attribute (one of ``FLOW``, ``BRANCH_TRUE``,
            ``BRANCH_FALSE``, ``LOOP_BACK``, ``DATA``).

        Raises:
            SyntaxError: If *source_code* cannot be parsed.
        """
        self._counter = 0
        self._graph = nx.DiGraph()
        self._var_defs = {}
        self._pending_edge_types = {}
        tree = ast.parse(source_code)
        entry = self._add_node("ENTRY", 0, "ENTRY")
        last_nodes = [entry]
        last_nodes = self._visit_stmts(tree.body, last_nodes)
        exit_node = self._add_node("EXIT", 0, "EXIT")
        for node in last_nodes:
            self._add_flow_edge(node, exit_node)
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

    def _add_flow_edge(self, src: int, dst: int, edge_type: str = "FLOW") -> None:
        """Add a typed control-flow edge from *src* to *dst*.

        If the edge already exists it is left unchanged to preserve any
        previously assigned type.
        """
        if not self._graph.has_edge(src, dst):
            self._graph.add_edge(src, dst, type=edge_type)

    def _add_data_edge(self, src: int, dst: int) -> None:
        """Add a DATA dependency edge from *src* to *dst* if none exists yet."""
        if not self._graph.has_edge(src, dst):
            self._graph.add_edge(src, dst, type="DATA")

    def _connect_from(self, prev_nodes: list[int], new_node: int) -> None:
        """Add typed edges from every predecessor to *new_node*.

        The edge type is taken from ``_pending_edge_types`` for each
        predecessor (defaulting to ``FLOW``).
        """
        for p in prev_nodes:
            edge_type = self._pending_edge_types.pop(p, "FLOW")
            self._add_flow_edge(p, new_node, edge_type)

    def _update_var_defs(self, node_id: int, var_names: Set[str]) -> None:
        """Record that *node_id* is the most recent definition of each name."""
        for name in var_names:
            self._var_defs[name] = node_id

    def _add_data_edges_for_used(self, current_node: int, used_vars: Set[str]) -> None:
        """Add DATA edges from the defining nodes of *used_vars* to *current_node*."""
        for var in used_vars:
            def_node = self._var_defs.get(var)
            if def_node is not None and def_node != current_node:
                self._add_data_edge(def_node, current_node)

    @staticmethod
    def _get_used_names(node: ast.AST) -> Set[str]:
        """Return all ``Name`` ids referenced in *node* (reads)."""
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    @staticmethod
    def _get_assigned_names(targets: list[ast.expr]) -> Set[str]:
        """Return the variable names bound by *targets* (writes)."""
        names: Set[str] = set()
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.add(elt.id)
        return names

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

    def _handle_assign(self, stmt: ast.Assign, prev_nodes: list[int]) -> list[int]:
        targets = ", ".join(ast.unparse(t) for t in stmt.targets)
        label = f"ASSIGN: {targets} = {ast.unparse(stmt.value)}"
        node = self._add_node("ASSIGN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        # DATA edges for variables read on the right-hand side
        self._add_data_edges_for_used(node, self._get_used_names(stmt.value))
        # Record what this node defines
        self._update_var_defs(node, self._get_assigned_names(stmt.targets))
        return [node]

    def _handle_aug_assign(self, stmt: ast.AugAssign, prev_nodes: list[int]) -> list[int]:
        op_class = type(stmt.op).__name__
        op_str = _AUG_OP_MAP.get(op_class, "?=")
        label = f"AUG_ASSIGN: {ast.unparse(stmt.target)} {op_str} {ast.unparse(stmt.value)}"
        node = self._add_node("AUG_ASSIGN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        # Both the target (read-modify-write) and value are used
        used = self._get_used_names(stmt.value) | self._get_used_names(stmt.target)
        self._add_data_edges_for_used(node, used)
        if isinstance(stmt.target, ast.Name):
            self._update_var_defs(node, {stmt.target.id})
        return [node]

    def _handle_ann_assign(self, stmt: ast.AnnAssign, prev_nodes: list[int]) -> list[int]:
        label = f"ANN_ASSIGN: {ast.unparse(stmt.target)}: {ast.unparse(stmt.annotation)}"
        if stmt.value:
            label += f" = {ast.unparse(stmt.value)}"
        node = self._add_node("ANN_ASSIGN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        if stmt.value:
            self._add_data_edges_for_used(node, self._get_used_names(stmt.value))
        if isinstance(stmt.target, ast.Name):
            self._update_var_defs(node, {stmt.target.id})
        return [node]

    def _handle_if(self, stmt: ast.If, prev_nodes: list[int]) -> list[int]:
        cond_label = f"IF: {ast.unparse(stmt.test)}"
        cond_node = self._add_node("IF", stmt.lineno, cond_label)
        self._connect_from(prev_nodes, cond_node)
        # DATA edges for variables read in the condition
        self._add_data_edges_for_used(cond_node, self._get_used_names(stmt.test))

        # True branch — first edge from cond_node gets type BRANCH_TRUE
        self._pending_edge_types[cond_node] = "BRANCH_TRUE"
        true_exits = self._visit_stmts(stmt.body, [cond_node])

        # False / elif branch
        if stmt.orelse:
            self._pending_edge_types[cond_node] = "BRANCH_FALSE"
            false_exits = self._visit_stmts(stmt.orelse, [cond_node])
        else:
            # No else: cond_node is also an open exit (falls through on False)
            self._pending_edge_types[cond_node] = "BRANCH_FALSE"
            false_exits = [cond_node]

        return true_exits + false_exits

    def _handle_for(self, stmt: ast.For, prev_nodes: list[int]) -> list[int]:
        label = f"FOR: {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}"
        loop_node = self._add_node("FOR", stmt.lineno, label)
        self._connect_from(prev_nodes, loop_node)
        # DATA edges for the iterable expression
        self._add_data_edges_for_used(loop_node, self._get_used_names(stmt.iter))
        # Define the loop variable
        self._update_var_defs(loop_node, self._get_assigned_names([stmt.target]))

        body_exits = self._visit_stmts(stmt.body, [loop_node])
        # Back-edge from body to loop header
        for node in body_exits:
            self._add_flow_edge(node, loop_node, "LOOP_BACK")

        return [loop_node]  # exit edge from loop header

    def _handle_while(self, stmt: ast.While, prev_nodes: list[int]) -> list[int]:
        label = f"WHILE: {ast.unparse(stmt.test)}"
        loop_node = self._add_node("WHILE", stmt.lineno, label)
        self._connect_from(prev_nodes, loop_node)
        self._add_data_edges_for_used(loop_node, self._get_used_names(stmt.test))

        body_exits = self._visit_stmts(stmt.body, [loop_node])
        for node in body_exits:
            self._add_flow_edge(node, loop_node, "LOOP_BACK")

        return [loop_node]

    def _handle_funcdef(
        self,
        stmt: ast.FunctionDef | ast.AsyncFunctionDef,
        prev_nodes: list[int],
    ) -> list[int]:
        label = f"FUNCDEF: {stmt.name}({', '.join(a.arg for a in stmt.args.args)})"
        func_node = self._add_node("FUNCDEF", stmt.lineno, label)
        self._connect_from(prev_nodes, func_node)
        self._update_var_defs(func_node, {stmt.name})
        # Walk the function body connected from the func_def node
        self._visit_stmts(stmt.body, [func_node])
        return [func_node]

    def _handle_return(self, stmt: ast.Return, prev_nodes: list[int]) -> list[int]:
        value = ast.unparse(stmt.value) if stmt.value else "None"
        label = f"RETURN: {value}"
        node = self._add_node("RETURN", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        if stmt.value:
            self._add_data_edges_for_used(node, self._get_used_names(stmt.value))
        return [node]

    def _handle_expr(self, stmt: ast.Expr, prev_nodes: list[int]) -> list[int]:
        """Handle expression statements (typically function calls)."""
        if isinstance(stmt.value, ast.Call):
            label = f"CALL: {ast.unparse(stmt.value)}"
            node = self._add_node("CALL", stmt.lineno, label)
        else:
            label = f"EXPR: {ast.unparse(stmt.value)}"
            node = self._add_node("EXPR", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        self._add_data_edges_for_used(node, self._get_used_names(stmt.value))
        return [node]

    def _handle_import(self, stmt: ast.stmt, prev_nodes: list[int]) -> list[int]:
        label = f"IMPORT: {ast.unparse(stmt)}"
        node = self._add_node("IMPORT", stmt.lineno, label)
        self._connect_from(prev_nodes, node)
        # Define the imported module/name bindings
        if isinstance(stmt, ast.Import):
            defined = {(alias.asname or alias.name).split(".")[0] for alias in stmt.names}
        else:  # ast.ImportFrom
            defined = {alias.asname or alias.name for alias in stmt.names}
        self._update_var_defs(node, defined)
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
            graph: The PDG graph.
            node_id: Integer node identifier.

        Returns:
            Dictionary with at least ``type``, ``lineno``, and ``label`` keys.
        """
        return graph.nodes.get(node_id, {"type": "", "lineno": 0, "label": ""})

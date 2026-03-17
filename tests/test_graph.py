"""Unit tests for app/graph/builder.py and app/graph/comparator.py."""

from __future__ import annotations

import pytest
import networkx as nx

from app.graph.builder import GraphBuilder
from app.graph.comparator import GraphComparator


# ---------------------------------------------------------------------------
# GraphBuilder tests
# ---------------------------------------------------------------------------


class TestGraphBuilder:
    """Tests for GraphBuilder.build()."""

    def setup_method(self) -> None:
        self.builder = GraphBuilder()

    def test_simple_assignment(self) -> None:
        """Building a graph from a single assignment should produce nodes."""
        code = "x = 1"
        g = self.builder.build(code)
        assert isinstance(g, nx.DiGraph)
        types = [d["type"] for _, d in g.nodes(data=True)]
        assert "ASSIGN" in types
        assert "ENTRY" in types
        assert "EXIT" in types

    def test_if_statement(self) -> None:
        """An IF statement should produce an IF-type node."""
        code = (
            "x = 10\n"
            "if x > 5:\n"
            "    y = 1\n"
            "else:\n"
            "    y = 0\n"
        )
        g = self.builder.build(code)
        types = [d["type"] for _, d in g.nodes(data=True)]
        assert "IF" in types

    def test_for_loop(self) -> None:
        """A for loop should produce a FOR-type node."""
        code = (
            "total = 0\n"
            "for i in range(10):\n"
            "    total += i\n"
        )
        g = self.builder.build(code)
        types = [d["type"] for _, d in g.nodes(data=True)]
        assert "FOR" in types
        assert "AUG_ASSIGN" in types

    def test_while_loop(self) -> None:
        """A while loop should produce a WHILE-type node."""
        code = (
            "n = 10\n"
            "while n > 0:\n"
            "    n -= 1\n"
        )
        g = self.builder.build(code)
        types = [d["type"] for _, d in g.nodes(data=True)]
        assert "WHILE" in types

    def test_function_definition(self) -> None:
        """A function definition should produce a FUNCDEF-type node."""
        code = (
            "def add(a, b):\n"
            "    return a + b\n"
        )
        g = self.builder.build(code)
        types = [d["type"] for _, d in g.nodes(data=True)]
        assert "FUNCDEF" in types
        assert "RETURN" in types

    def test_function_call(self) -> None:
        """A bare function call expression should produce a CALL-type node."""
        code = "print('hello')\n"
        g = self.builder.build(code)
        types = [d["type"] for _, d in g.nodes(data=True)]
        assert "CALL" in types

    def test_entry_exit_nodes(self) -> None:
        """Every graph must have exactly one ENTRY and one EXIT node."""
        code = "pass"
        g = self.builder.build(code)
        entry_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "ENTRY"]
        exit_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "EXIT"]
        assert len(entry_nodes) == 1
        assert len(exit_nodes) == 1

    def test_node_attributes(self) -> None:
        """Every node must carry 'type', 'lineno', and 'label' attributes."""
        code = "x = 42\n"
        g = self.builder.build(code)
        for _, attrs in g.nodes(data=True):
            assert "type" in attrs
            assert "lineno" in attrs
            assert "label" in attrs

    def test_empty_code(self) -> None:
        """Building a graph from empty code should not raise an exception."""
        g = self.builder.build("")
        assert isinstance(g, nx.DiGraph)

    def test_syntax_error_raises(self) -> None:
        """Invalid Python should raise SyntaxError."""
        with pytest.raises(SyntaxError):
            self.builder.build("def broken(:\n    pass\n")

    def test_nested_if_for(self) -> None:
        """Nested constructs should produce multiple node types."""
        code = (
            "for i in range(5):\n"
            "    if i % 2 == 0:\n"
            "        print(i)\n"
        )
        g = self.builder.build(code)
        types = set(d["type"] for _, d in g.nodes(data=True))
        assert "FOR" in types
        assert "IF" in types

    def test_graph_has_edges(self) -> None:
        """A non-trivial program should produce a connected graph."""
        code = (
            "x = 1\n"
            "y = 2\n"
            "z = x + y\n"
        )
        g = self.builder.build(code)
        assert g.number_of_edges() > 0


# ---------------------------------------------------------------------------
# GraphComparator tests
# ---------------------------------------------------------------------------


class TestGraphComparator:
    """Tests for GraphComparator.compare()."""

    def setup_method(self) -> None:
        self.builder = GraphBuilder()
        self.comparator = GraphComparator()

    def test_identical_code_similarity_one(self) -> None:
        """Identical code should yield similarity_score == 1.0."""
        code = (
            "x = 1\n"
            "y = x + 2\n"
        )
        g1 = self.builder.build(code)
        g2 = self.builder.build(code)
        result = self.comparator.compare(g1, g2)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-4)

    def test_different_code_similarity_less_than_one(self) -> None:
        """Different code should yield similarity_score < 1.0."""
        ref = "x = 1\nfor i in range(10):\n    x += i\n"
        stu = "x = 0\n"
        g_ref = self.builder.build(ref)
        g_stu = self.builder.build(stu)
        result = self.comparator.compare(g_ref, g_stu)
        assert result.similarity_score < 1.0

    def test_missing_nodes_detected(self) -> None:
        """Missing FOR node should be reported."""
        ref = "for i in range(5):\n    print(i)\n"
        stu = "print('done')\n"
        g_ref = self.builder.build(ref)
        g_stu = self.builder.build(stu)
        result = self.comparator.compare(g_ref, g_stu)
        missing_types = [n["type"] for n in result.missing_nodes]
        assert "FOR" in missing_types

    def test_extra_nodes_detected(self) -> None:
        """Extra FOR node in student code should be reported."""
        ref = "x = 1\n"
        stu = "x = 1\nfor i in range(3):\n    pass\n"
        g_ref = self.builder.build(ref)
        g_stu = self.builder.build(stu)
        result = self.comparator.compare(g_ref, g_stu)
        extra_types = [n["type"] for n in result.extra_nodes]
        assert "FOR" in extra_types

    def test_similarity_score_range(self) -> None:
        """Similarity score must always be in [0.0, 1.0]."""
        ref = "x = 1\nif x:\n    y = 2\n"
        stu = "a = 10\nwhile a:\n    a -= 1\n"
        g_ref = self.builder.build(ref)
        g_stu = self.builder.build(stu)
        result = self.comparator.compare(g_ref, g_stu)
        assert 0.0 <= result.similarity_score <= 1.0

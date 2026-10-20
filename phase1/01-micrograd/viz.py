"""Visualization helpers for micrograd computation graphs."""

from graphviz import Digraph


def trace(root):
    """Collect all nodes and edges reachable from root via _prev."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, rankdir="LR"):
    """Render the graph ending at root. Each Value shows data and grad.

    Works with any object exposing .data, .grad, ._prev and ._op.
    An optional .label attribute is shown if present.
    """
    dot = Digraph(format="svg", graph_attr={"rankdir": rankdir})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        label = getattr(n, "label", "")
        prefix = f"{label} | " if label else ""
        dot.node(
            name=uid,
            label=f"{{ {prefix}data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for child, parent in edges:
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot

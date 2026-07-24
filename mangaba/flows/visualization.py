"""
Flow visualisation — self-contained HTML/SVG rendering of a flow graph.

The output is a single HTML file with inline CSS and inline SVG: no CDN, no
JavaScript library, nothing to download. Methods are laid out in layers
(entry points on top, downstream listeners below), route labels appear as
pill-shaped nodes, and loop-backs are drawn as dashed side edges.
"""

from __future__ import annotations

import html
import logging
import os
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from mangaba.core.exceptions import FlowError
from mangaba.flows.flow import FlowEdge, FlowGraph, FlowNode

if TYPE_CHECKING:  # pragma: no cover
    from mangaba.flows.flow import Flow

log = logging.getLogger(__name__)

# Layout constants (SVG user units == CSS pixels).
_NODE_W = 190
_NODE_H = 56
_LABEL_W = 130
_LABEL_H = 34
_GAP_X = 46
_GAP_Y = 96
_PAD = 40

_PALETTE = {
    "start": ("#e8f5e9", "#2e7d32", "#1b5e20"),
    "listener": ("#e8f0fe", "#1565c0", "#0d47a1"),
    "router": ("#fff4e5", "#ef6c00", "#e65100"),
    "label": ("#f3e8fd", "#7b1fa2", "#4a148c"),
}

_KIND_TITLES = {
    "start": "entry point",
    "listener": "listener",
    "router": "router",
    "label": "route label",
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def compute_layers(graph: FlowGraph, roots: List[str]) -> Dict[str, int]:
    """Assign a depth to every node via longest-path relaxation.

    Args:
        graph: The flow graph to lay out.
        roots: Entry-point node names, pinned to layer 0.

    Returns:
        Mapping of node name to layer index.
    """
    layers: Dict[str, int] = {node.name: 0 for node in graph.nodes}
    root_set = set(roots)

    for _ in range(len(graph.nodes) + 1):
        changed = False
        for edge in graph.edges:
            if edge.target in root_set:
                continue  # loop-back into an entry point
            candidate = layers.get(edge.source, 0) + 1
            if candidate > layers.get(edge.target, 0):
                layers[edge.target] = candidate
                changed = True
        if not changed:
            break
    return layers


def _positions(graph: FlowGraph, layers: Dict[str, int]) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
    """Return node centre positions plus the overall canvas size."""
    by_layer: Dict[int, List[FlowNode]] = {}
    for node in graph.nodes:
        by_layer.setdefault(layers.get(node.name, 0), []).append(node)

    widest = 1
    for nodes in by_layer.values():
        widest = max(widest, len(nodes))

    width = _PAD * 2 + widest * _NODE_W + (widest - 1) * _GAP_X
    depth = max(by_layer) if by_layer else 0
    height = _PAD * 2 + (depth + 1) * _NODE_H + depth * _GAP_Y

    centres: Dict[str, Tuple[float, float]] = {}
    for layer, nodes in by_layer.items():
        row_width = len(nodes) * _NODE_W + (len(nodes) - 1) * _GAP_X
        left = (width - row_width) / 2
        for index, node in enumerate(nodes):
            cx = left + index * (_NODE_W + _GAP_X) + _NODE_W / 2
            cy = _PAD + layer * (_NODE_H + _GAP_Y) + _NODE_H / 2
            centres[node.name] = (cx, cy)
    return centres, int(width), int(height)


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _node_box(node: FlowNode) -> Tuple[int, int]:
    return (_LABEL_W, _LABEL_H) if node.kind == "label" else (_NODE_W, _NODE_H)


def _render_node(node: FlowNode, cx: float, cy: float) -> str:
    fill, stroke, text_colour = _PALETTE.get(node.kind, _PALETTE["listener"])
    box_w, box_h = _node_box(node)
    x = cx - box_w / 2
    y = cy - box_h / 2
    radius = box_h / 2 if node.kind == "label" else 10

    tooltip = f"{node.name} — {_KIND_TITLES.get(node.kind, node.kind)}"
    if node.condition:
        tooltip += f"\ntriggered by {node.condition}"
    if node.doc:
        tooltip += f"\n{node.doc}"

    parts = [
        f'<g class="node node--{node.kind}">',
        f'<title>{html.escape(tooltip)}</title>',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w}" height="{box_h}" rx="{radius:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2" />',
    ]

    if node.kind == "label":
        parts.append(
            f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" class="t-label" '
            f'fill="{text_colour}">{html.escape(_truncate(node.name, 16))}</text>'
        )
    else:
        badge = _KIND_TITLES.get(node.kind, node.kind)
        if node.is_async:
            badge += " · async"
        parts.append(
            f'<text x="{cx:.1f}" y="{cy - 3:.1f}" text-anchor="middle" class="t-name" '
            f'fill="{text_colour}">{html.escape(_truncate(node.name, 22))}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{cy + 15:.1f}" text-anchor="middle" class="t-kind" '
            f'fill="{stroke}">{html.escape(badge)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _render_edge(
    edge: FlowEdge,
    centres: Dict[str, Tuple[float, float]],
    boxes: Dict[str, Tuple[int, int]],
    canvas_width: int,
) -> str:
    if edge.source not in centres or edge.target not in centres:
        return ""

    sx, sy = centres[edge.source]
    tx, ty = centres[edge.target]
    sh = boxes[edge.source][1] / 2
    th = boxes[edge.target][1] / 2

    dashed = edge.kind == "route"
    css_class = f"edge edge--{edge.kind}"

    if ty <= sy:
        # Loop-back: route around the right-hand side.
        detour = min(canvas_width - 12, max(sx, tx) + _NODE_W / 2 + 34)
        path = (
            f"M {sx:.1f} {sy + sh:.1f} "
            f"C {detour:.1f} {sy + sh + 30:.1f}, {detour:.1f} {ty - th - 30:.1f}, "
            f"{tx:.1f} {ty - th:.1f}"
        )
        css_class += " edge--loop"
    else:
        midpoint = (sy + sh + ty - th) / 2
        path = (
            f"M {sx:.1f} {sy + sh:.1f} "
            f"C {sx:.1f} {midpoint:.1f}, {tx:.1f} {midpoint:.1f}, {tx:.1f} {ty - th:.1f}"
        )

    dash = ' stroke-dasharray="6 5"' if dashed else ""
    svg = (
        f'<path class="{css_class}" d="{path}" fill="none"{dash} marker-end="url(#arrow)" />'
    )

    marker = {"and": "AND", "or": "OR"}.get(edge.kind, "")
    if marker:
        mx = (sx + tx) / 2
        my = (sy + sh + ty - th) / 2
        svg += (
            f'<rect x="{mx - 17:.1f}" y="{my - 11:.1f}" width="34" height="18" rx="9" class="joinbox" />'
            f'<text x="{mx:.1f}" y="{my + 2:.1f}" text-anchor="middle" class="t-join">{marker}</text>'
        )
    return svg


def render_graph_html(graph: FlowGraph, roots: List[str], title: str = "") -> str:
    """Render a :class:`FlowGraph` to a standalone HTML document.

    Args:
        graph: Graph produced by ``flow.graph``.
        roots: Entry-point method names (drawn on the first layer).
        title: Page heading; defaults to the graph name.

    Returns:
        The full HTML document as a string.

    Example::

        html_text = render_graph_html(flow.graph, ["begin"], "My Flow")
    """
    heading = title or graph.name or "Flow"
    if not graph.nodes:
        body_svg = '<p class="empty">This flow has no decorated methods yet.</p>'
        width = 640
    else:
        layers = compute_layers(graph, roots)
        centres, width, height = _positions(graph, layers)
        boxes = {node.name: _node_box(node) for node in graph.nodes}

        edges_svg = "".join(
            _render_edge(edge, centres, boxes, width) for edge in graph.edges
        )
        nodes_svg = "".join(
            _render_node(node, *centres[node.name]) for node in graph.nodes
        )
        body_svg = (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{html.escape(heading)} graph" '
            f'xmlns="http://www.w3.org/2000/svg">'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8a94a6" /></marker></defs>'
            f"{edges_svg}{nodes_svg}</svg>"
        )

    counts = {kind: 0 for kind in _PALETTE}
    for node in graph.nodes:
        counts[node.kind] = counts.get(node.kind, 0) + 1

    legend = "".join(
        f'<li><span class="swatch" style="background:{_PALETTE[kind][0]};'
        f'border-color:{_PALETTE[kind][1]}"></span>{_KIND_TITLES[kind]} ({counts.get(kind, 0)})</li>'
        for kind in ("start", "router", "listener", "label")
    )

    return _TEMPLATE.format(
        title=html.escape(heading),
        legend=legend,
        svg=body_svg,
        edge_count=len(graph.edges),
        node_count=len(graph.nodes),
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title} — Mangaba Flow</title>
<style>
  :root {{
    --bg: #f7f8fa; --fg: #1f2430; --muted: #667085; --card: #ffffff; --line: #e3e6ec;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #14161c; --fg: #e8eaf0; --muted: #98a2b3; --card: #1c1f27; --line: #2b303b; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 28px 20px 48px;
    background: var(--bg); color: var(--fg);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  header {{ max-width: 1080px; margin: 0 auto 18px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); font-size: 13px; }}
  .canvas {{
    max-width: 1080px; margin: 0 auto; padding: 20px; overflow-x: auto;
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  }}
  svg {{ display: block; margin: 0 auto; max-width: 100%; height: auto; }}
  .edge {{ stroke: #8a94a6; stroke-width: 2; }}
  .edge--route {{ stroke: #a06bd0; }}
  .edge--loop {{ stroke: #b0b7c3; }}
  .joinbox {{ fill: #fff; stroke: #8a94a6; stroke-width: 1.5; }}
  .t-name {{ font-size: 14px; font-weight: 600; }}
  .t-kind {{ font-size: 10.5px; letter-spacing: 0.04em; text-transform: uppercase; }}
  .t-label {{ font-size: 12.5px; font-weight: 600; }}
  .t-join {{ font-size: 10px; font-weight: 700; fill: #55607a; }}
  .node {{ cursor: default; }}
  ul.legend {{
    max-width: 1080px; margin: 16px auto 0; padding: 0; list-style: none;
    display: flex; flex-wrap: wrap; gap: 18px; color: var(--muted); font-size: 13px;
  }}
  ul.legend li {{ display: flex; align-items: center; gap: 7px; }}
  .swatch {{ width: 14px; height: 14px; border-radius: 4px; border: 2px solid; display: inline-block; }}
  .empty {{ color: var(--muted); text-align: center; padding: 40px 0; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">{node_count} nodes · {edge_count} connections · generated by Mangaba AI Flows</div>
</header>
<main class="canvas">{svg}</main>
<ul class="legend">{legend}</ul>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plot_flow(flow: Any, filename: str = "flow.html") -> str:
    """Write a flow's graph to a self-contained HTML file.

    Args:
        flow: A :class:`~mangaba.flows.flow.Flow` instance.
        filename: Destination path; ``.html`` is appended when missing.

    Returns:
        Absolute path of the written file.

    Raises:
        FlowError: If the file cannot be written.

    Example::

        plot_flow(MyFlow(), "docs/my_flow.html")
    """
    graph = getattr(flow, "graph", None)
    if not isinstance(graph, FlowGraph):
        raise FlowError(f"{type(flow).__name__} is not a Flow — nothing to plot")

    if not filename.lower().endswith((".html", ".htm")):
        filename = f"{filename}.html"
    path = os.path.abspath(filename)

    roots = list(getattr(flow, "_start_methods", []))
    document = render_graph_html(graph, roots, title=graph.name)

    parent = os.path.dirname(path)
    try:
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(document)
    except OSError as exc:
        raise FlowError(f"Could not write flow plot to '{path}': {exc}", cause=exc) from exc

    log.info("Flow plot written to %s", path)
    return path

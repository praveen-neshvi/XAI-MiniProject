"""
PyVis HTML for GNNExplainer on a MUTAG ego graph.

Full ego graph with top-k nodes and edges highlighted; everything else grayed out.

Example:
  python visualize_explanation.py d305 --top-k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mutag_ego_graphs import EgoGraph

from visualize_ego_graph import (
    ROLE_COLORS,
    _literal_hint,
    _type_targets,
    build_ego_for_molecule,
    node_role,
    short_label,
    short_predicate,
)

CENTER_COLOR = "#e74c3c"
CENTER_SIZE = 28
DEFAULT_NODE_SIZE = 14
MUTED_NODE_COLOR = "#d0d4d8"
MUTED_EDGE_COLOR = "#e2e5e8"
IMPORTANT_EDGE_COLOR = "#2c3e50"
TOP_EDGE_WIDTH = 4
MUTED_EDGE_WIDTH = 1


def _top_k_node_set(g: EgoGraph, ranked_nodes: list[dict], top_k: int) -> set[str]:
    """Top-k masked nodes only (molecule center is shown separately in red)."""
    nodes: set[str] = set()
    for row in ranked_nodes:
        if row.get("is_root") or row["node"] == g.center:
            continue
        nodes.add(row["node"])
        if len(nodes) >= top_k:
            break
    return nodes


def _important_triple_scores(
    g: EgoGraph,
    top_edges: list[dict],
    top_k: int,
) -> dict[tuple[str, str, str], float]:
    """Map RDF triple (s, p, o) -> importance for top-k explainer edges."""
    scores: dict[tuple[str, str, str], float] = {}
    for row in top_edges[:top_k]:
        pred = row["predicate"]
        score = float(row["importance"])
        if pred.endswith(" (reverse)"):
            subj, obj = row["object"], row["subject"]
            pred_short = pred[: -len(" (reverse)")]
        else:
            subj, obj = row["subject"], row["object"]
            pred_short = pred

        for s, p, o in g.resource_triples:
            if s == subj and o == obj and short_predicate(p) == pred_short:
                key = (s, p, o)
                scores[key] = max(scores.get(key, 0.0), score)
                break
    return scores


def explained_ego_to_pyvis(
    g: EgoGraph,
    ranked_nodes: list[dict],
    ranked_edges: list[dict],
    *,
    top_k: int = 10,
    target_class: int | None = None,
    label_names: dict[int, str] | None = None,
    height: str = "750px",
    width: str = "100%",
):
    from pyvis.network import Network

    label_names = label_names or {0: "non-mutagenic", 1: "mutagenic"}
    top_k_nodes = _top_k_node_set(g, ranked_nodes, top_k)
    node_scores = {row["node"]: float(row["importance"]) for row in ranked_nodes}
    important_triples = _important_triple_scores(g, ranked_edges, top_k)

    net = Network(height=height, width=width, directed=True, notebook=False)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)
    type_targets = _type_targets(g)

    for node in sorted(g.nodes, key=short_label):
        is_center = node == g.center
        is_important = node in top_k_nodes
        role = node_role(node, g, type_targets)

        title = node
        hint = _literal_hint(g, node)
        if hint:
            title += f"\n{hint}"
        if is_important and node in node_scores:
            title += f"\nimportance={node_scores[node]:.4f}"
        elif not is_important:
            title += "\n(not in top-k nodes)"

        if is_center:
            color = CENTER_COLOR
            size = CENTER_SIZE
            border = 3
            title += "\n(molecule center — prediction target, not node-masked)"
        elif is_important:
            color = ROLE_COLORS.get(role, "#bdc3c7")
            size = DEFAULT_NODE_SIZE
            border = 2
        else:
            color = MUTED_NODE_COLOR
            size = DEFAULT_NODE_SIZE
            border = 1

        net.add_node(
            node,
            label=short_label(node),
            title=title,
            color=color,
            size=size,
            borderWidth=border,
        )

    seen: set[tuple[str, str, str]] = set()
    for s, p, o in g.resource_triples:
        key = (s, p, o)
        if key in seen:
            continue
        seen.add(key)
        pred_short = short_predicate(p)

        if key in important_triples:
            score = important_triples[key]
            title = f"{pred_short}\nimportance={score:.4f}"
            net.add_edge(
                s,
                o,
                label="",
                title=title,
                arrows="to",
                width=TOP_EDGE_WIDTH,
                color=IMPORTANT_EDGE_COLOR,
            )
        else:
            net.add_edge(
                s,
                o,
                label="",
                title=f"{pred_short}\n(not in top-{top_k} edges)",
                arrows="to",
                width=MUTED_EDGE_WIDTH,
                color=MUTED_EDGE_COLOR,
                dashes=[10, 8],
            )

    target_txt = ""
    if target_class is not None:
        target_txt = f", explained={label_names.get(target_class, target_class)}"

    chart_title = (
        f"GNNExplainer — {short_label(g.center)} "
        f"(top-{top_k} nodes & edges{target_txt})"
    )
    net.set_options(
        """
    {
      "nodes": {"font": {"size": 14}},
      "edges": {
        "font": {"size": 0, "align": "middle"},
        "smooth": {"type": "dynamic"}
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150}
      }
    }
    """
    )
    return net, chart_title


def write_explained_ego_html(
    g: EgoGraph,
    ranked_nodes: list[dict],
    ranked_edges: list[dict],
    output_path: Path,
    *,
    top_k: int = 10,
    target_class: int | None = None,
    label_names: dict[int, str] | None = None,
) -> None:
    net, title = explained_ego_to_pyvis(
        g,
        ranked_nodes,
        ranked_edges,
        top_k=top_k,
        target_class=target_class,
        label_names=label_names,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)

    html = output_path.read_text(encoding="utf-8")
    legend = (
        "<p style='font-family:sans-serif;padding:0 12px;color:#444'>"
        "<b>Red</b> = molecule center (explained target, not node-masked). "
        "<b>Colored nodes</b> = top-k important nodes. "
        "<b>Thick dark edges</b> = top-k important edges (hover for score). "
        "<b>Gray</b> = remaining ego graph context."
        "</p>\n"
    )
    injected = f"<h2 style='font-family:sans-serif;padding:8px 12px'>{title}</h2>\n{legend}"
    if "<body>" in html:
        html = html.replace("<body>", f"<body>\n{injected}", 1)
    else:
        html = injected + html
    output_path.write_text(html, encoding="utf-8")


# Backwards-compatible alias used by explain_rgcn.py
write_explained_top_edges_html = write_explained_ego_html


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize GNNExplainer on MUTAG ego graph.")
    parser.add_argument("molecule", help="Molecule id, e.g. d305")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--mutag-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Datasets" / "mutag-hetero",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "rgcn_explainable.pt",
    )
    parser.add_argument("--explainer-epochs", type=int, default=200)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    from explain_rgcn import LABEL_NAMES, explain_molecule

    result = explain_molecule(
        args.molecule,
        model_path=args.model,
        mutag_dir=args.mutag_dir,
        explainer_epochs=args.explainer_epochs,
        top_k=args.top_k,
    )
    g = build_ego_for_molecule(args.mutag_dir, args.molecule)
    out = args.output or (
        Path(__file__).resolve().parent
        / "visualizations"
        / f"explain_{args.molecule}_graph.html"
    )
    write_explained_ego_html(
        g,
        result["ranked_nodes"],
        result["ranked_edges"],
        out,
        top_k=args.top_k,
        target_class=result["target_class"],
        label_names=LABEL_NAMES,
    )
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

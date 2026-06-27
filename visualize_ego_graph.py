"""
Visualize MUTAG ego graphs as interactive HTML (PyVis).

Example:
  python visualize_ego_graph.py d305
  python visualize_ego_graph.py --uri http://dl-learner.org/carcinogenesis#d187 -o ego_d187.html
  python visualize_ego_graph.py --list
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from mutag_ego_graphs import (
    RDF_TYPE,
    EgoGraph,
    build_molecule_indices,
    ego_graph_for_molecule,
    load_labeled_molecules,
    uri_to_nt,
)

CARCIN_PREFIX = "http://dl-learner.org/carcinogenesis#"
COMPOUND = f"<{CARCIN_PREFIX}Compound>"


def short_label(uri: str) -> str:
    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def short_predicate(pred: str) -> str:
    frag = short_label(pred)
    if frag == "type":
        return "rdf:type"
    return frag


def node_role(node: str, g: EgoGraph, type_targets: set[str]) -> str:
    if node == g.center:
        return "compound"
    if node in type_targets:
        return "class"
    frag = short_label(node)
    if frag.startswith("bond"):
        return "bond"
    if re.match(r"^d\d+_\d+", frag):
        return "atom"
    if any(x in frag.lower() for x in ("ring", "ketone", "ester", "ether", "amino", "methyl")):
        return "structure"
    return "other"


def _type_targets(g: EgoGraph) -> set[str]:
    return {o for s, p, o in g.resource_triples if p == RDF_TYPE}


def _literal_hint(g: EgoGraph, node: str, max_items: int = 2) -> str:
    hints: list[str] = []
    for s, p, lit in g.literal_triples:
        if s != node:
            continue
        pred = short_predicate(p)
        lit_short = lit.replace('"', "").split("^^")[0][:20]
        hints.append(f"{pred}={lit_short}")
        if len(hints) >= max_items:
            break
    return "; ".join(hints)


ROLE_COLORS = {
    "compound": "#e74c3c",
    "atom": "#3498db",
    "bond": "#2ecc71",
    "structure": "#9b59b6",
    "class": "#95a5a6",
    "other": "#f39c12",
}


def ego_graph_to_pyvis(g: EgoGraph, height: str = "750px", width: str = "100%"):
    from pyvis.network import Network

    net = Network(height=height, width=width, directed=True, notebook=False)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)
    type_targets = _type_targets(g)

    for node in sorted(g.nodes, key=short_label):
        role = node_role(node, g, type_targets)
        title = node
        hint = _literal_hint(g, node)
        if hint:
            title += f"\n{hint}"
        label = short_label(node)
        net.add_node(
            node,
            label=label,
            title=title,
            color=ROLE_COLORS.get(role, "#bdc3c7"),
            size=28 if node == g.center else 14,
            borderWidth=3 if node == g.center else 1,
        )

    seen_edges: set[tuple[str, str, str]] = set()
    for s, p, o in g.resource_triples:
        key = (s, short_predicate(p), o)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        net.add_edge(s, o, label=short_predicate(p), title=p, arrows="to")

    title = (
        f"MUTAG ego graph — {short_label(g.center)} "
        f"(label={g.label}, nodes={len(g.nodes)}, edges={len(seen_edges)})"
    )
    net.set_options(
        """
    {
      "nodes": {"font": {"size": 14}},
      "edges": {
        "font": {"size": 10, "align": "middle"},
        "smooth": {"type": "dynamic"}
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150}
      }
    }
    """
    )
    return net, title


def build_ego_for_molecule(mutag_dir: Path, molecule: str) -> EgoGraph:
    if molecule.startswith("http"):
        center = uri_to_nt(molecule)
    elif molecule.startswith("<"):
        center = molecule
    else:
        center = uri_to_nt(f"{CARCIN_PREFIX}{molecule}")

    labels = load_labeled_molecules(mutag_dir / "completeDataset.tsv")
    if center not in labels:
        raise ValueError(
            f"Unknown molecule '{molecule}'. Use --list to see ids from completeDataset.tsv."
        )

    out_adj, inc_adj, literal_out = build_molecule_indices(mutag_dir / "mutag_stripped.nt")
    return ego_graph_for_molecule(center, labels[center], out_adj, inc_adj, literal_out)


def write_ego_html(g: EgoGraph, output_path: Path) -> None:
    net, title = ego_graph_to_pyvis(g)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)

    html = output_path.read_text(encoding="utf-8")
    injected = f"<h2 style='font-family:sans-serif;padding:8px 12px'>{title}</h2>\n"
    if "<body>" in html:
        html = html.replace("<body>", f"<body>\n{injected}", 1)
    else:
        html = injected + html
    output_path.write_text(html, encoding="utf-8")


def list_molecules(mutag_dir: Path, limit: int = 20) -> None:
    labels = load_labeled_molecules(mutag_dir / "completeDataset.tsv")
    print(f"Labeled compounds: {len(labels)}")
    for i, (uri, y) in enumerate(sorted(labels.items(), key=lambda x: short_label(x[0]))):
        if i >= limit:
            print(f"  ... and {len(labels) - limit} more")
            break
        print(f"  {short_label(uri):12s}  label={y}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a MUTAG ego graph with PyVis.")
    parser.add_argument(
        "molecule",
        nargs="?",
        help="Molecule id (e.g. d305) or full URI",
    )
    parser.add_argument("--uri", help="Full molecule URI (alternative to positional id)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (default: visualizations/ego_<id>.html)",
    )
    parser.add_argument(
        "--mutag-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Datasets" / "mutag-hetero",
    )
    parser.add_argument("--list", action="store_true", help="List labeled molecule ids and exit")
    parser.add_argument("--open", action="store_true", help="Open HTML in default browser after export")
    args = parser.parse_args()

    mutag_dir = args.mutag_dir
    if args.list:
        list_molecules(mutag_dir)
        return 0

    mol = args.uri or args.molecule
    if not mol:
        parser.error("Provide a molecule id (e.g. d305), --uri, or use --list")

    g = build_ego_for_molecule(mutag_dir, mol)
    if not g.nodes:
        print(f"No nodes in ego graph for {mol}")
        return 1

    out = args.output or (Path("visualizations") / f"ego_{short_label(g.center)}.html")
    write_ego_html(g, out)
    print(f"Wrote {out.resolve()}")
    print(f"  nodes={len(g.nodes)}, resource_edges={len(g.resource_triples)}, label={g.label}")
    print("  Legend: red=compound, blue=atom, green=bond, purple=structure, gray=class")

    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

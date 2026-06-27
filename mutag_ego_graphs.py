"""
Build molecule-centered ego graphs from MUTAG RDF (.nt) for all labeled compounds.

Only molecules listed in completeDataset.tsv (bond URI + label_mutagenic) get a
graph. Atoms, bonds, structure instances, etc. appear as nodes when reachable
from that molecule without traversing *outward* from shared rdf:type class hubs
(otherwise every compound would merge through Carbon-*, Ketone, ...).
"""

from __future__ import annotations

import csv
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"


@dataclass
class EgoGraph:                                                                   #Subgraph structure for one labeled molecule and its induced RDF neighborhood.
    center: str
    label: int
    nodes: set[str] = field(default_factory=set)
    resource_triples: list[tuple[str, str, str]] = field(default_factory=list)
    literal_triples: list[tuple[str, str, str]] = field(default_factory=list)


def uri_to_nt(uri: str) -> str:
    if uri.startswith("<") and uri.endswith(">"):
        return uri
    return f"<{uri}>"


def load_labeled_molecules(complete_dataset_tsv: Path) -> dict[str, int]:           #Map molecule URI (angle-bracket form) -> label (0/1)
    """Map molecule URI (angle-bracket form) -> label 0/1."""
    labels: dict[str, int] = {}
    with complete_dataset_tsv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            bond = row["bond"].strip()
            y = int(row["label_mutagenic"])
            labels[uri_to_nt(bond)] = y
    return labels


def _parse_nt_object(obj: str) -> tuple[str | None, bool]:                         #Return (resource_token, is_literal). Resource_token is like <http:...> or _:b0 ; literals are kept as raw obj string.
    obj = obj.strip()
    if obj.startswith("<") and obj.endswith(">"):
        return obj, False
    if obj.startswith("_:"):
        return obj, False
    return obj, True


_TRIPLE_RE = re.compile(
    r"^(?P<s><[^>]*>|_:[^\s]+)\s+(?P<p><[^>]*>)\s+(?P<o>.+?)\s*\.\s*$"
)                                                                                #split each .nt line into subject, predicate, and object strings


def iter_nt_triples(nt_path: Path):                                             #Yield (subject, predicate, object_raw) for each line; object may be literal.
    """Yield (subject, predicate, object_raw) for each line; object may be literal."""
    with nt_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _TRIPLE_RE.match(line)
            if not m:
                continue
            yield m.group("s"), m.group("p"), m.group("o").strip()


def build_molecule_indices(nt_path: Path):
    """
    out_adj[u] = list of (p, o) for resource objects o.
    inc_adj[u] = list of (p, s) for resource subjects s (triple s p u).
    literal_out[u] = list of (p, o_raw) for literal objects.
    """
    out_adj: dict[str, list[tuple[str, str]]] = {}
    inc_adj: dict[str, list[tuple[str, str]]] = {}
    literal_out: dict[str, list[tuple[str, str]]] = {}

    def add_out(s: str, p: str, o: str) -> None:
        out_adj.setdefault(s, []).append((p, o))

    def add_inc(u: str, p: str, s: str) -> None:
        inc_adj.setdefault(u, []).append((p, s))

    for s, p, o_raw in iter_nt_triples(nt_path):
        o_res, is_lit = _parse_nt_object(o_raw)
        if is_lit:
            literal_out.setdefault(s, []).append((p, o_raw))
            continue
        assert o_res is not None
        add_out(s, p, o_res)
        add_inc(o_res, p, s)

    return out_adj, inc_adj, literal_out


def ego_graph_for_molecule(
    center: str,
    label: int,
    out_adj: dict[str, list[tuple[str, str]]],
    inc_adj: dict[str, list[tuple[str, str]]],
    literal_out: dict[str, list[tuple[str, str]]],
) -> EgoGraph:
    """
    BFS over the RDF graph from ``center``.

    Nodes reached as objects of rdf:type (ontology / class nodes shared across the
    whole KB) are kept in the subgraph but are not expanded further, so other
    molecules' atoms are not pulled in.
    """
    seen: set[str] = set()                     #Tracks visited nodes
    expandable: dict[str, bool] = {}                #Tracks nodes that can be expanded (rdf:type nodes are kept but not expanded)
    q: deque[str] = deque()                    #Standard Queue for BFS

    if center not in out_adj and center not in inc_adj:                            #If the node has no connections, return an empty ego graph.
        return EgoGraph(center=center, label=label, nodes=set())

    # Start BFS from center node
    seen.add(center)
    expandable[center] = True
    q.append(center)

    while q:
        u = q.popleft()
        if not expandable.get(u, False):               #If 'u' is not expandable, skip it. Or if 'u' doesn't exist in expandable, assume False and skip.
            continue

        for p, o in out_adj.get(u, ()):                        #Explore outgoing edges from 'u'
            if o not in seen:
                seen.add(o)                                         #Mark 'o' as visited
                expandable[o] = False if p == RDF_TYPE else True    #If 'o' is a rdf:type node, it's not expandable. Otherwise, it is expandable.
                q.append(o)                                         #Add 'o' to the queue for further exploration

        for p, s in inc_adj.get(u, ()):                            #Explore incoming edges to 'u'
            if s not in seen:
                seen.add(s)                                     #Mark 's' as visited
                if p == RDF_TYPE:
                    expandable[s] = True                         #If 's' is a rdf:type node, it's expandable.
                else:
                    expandable[s] = True                         #If 's' is not a rdf:type node, it's expandable.
                q.append(s)                                     #Add 's' to the queue for further exploration

    nodes = seen                                                #The set of nodes in the ego graph
    res_triples: list[tuple[str, str, str]] = []                #List of resource triples in the ego graph
    lit_triples: list[tuple[str, str, str]] = []                #List of literal triples in the ego graph

    for s in nodes:                                    #Construct the ego graph by iterating over the nodes and adding the resource and literal triples.
        for p, o in out_adj.get(s, ()):
            if o in nodes:
                res_triples.append((s, p, o))
        for p, lit_raw in literal_out.get(s, ()):
            lit_triples.append((s, p, lit_raw))

    return EgoGraph(
        center=center,
        label=label,
        nodes=nodes,
        resource_triples=res_triples,
        literal_triples=lit_triples,
    )

    # Example of ego graph:
    # {
    #     "center": "<http://dl-learner.org/carcinogenesis#d305>",
    #     "label": 0,
    #     "nodes": {"<http://dl-learner.org/carcinogenesis#d305>", "<http://dl-learner.org/carcinogenesis#d305_24>"},
    #     "resource_triples": [("<http://dl-learner.org/carcinogenesis#d305>", "<http://dl-learner.org/carcinogenesis#hasAtom>", "<http://dl-learner.org/carcinogenesis#d305_24>")],
    #     "literal_triples": []
    # }
    

def build_all_ego_graphs(
    mutag_dir: Path,
    *,
    nt_name: str = "mutag_stripped.nt",
    labels_name: str = "completeDataset.tsv",
) -> list[EgoGraph]:
    complete = mutag_dir / labels_name
    nt_path = mutag_dir / nt_name
    labels = load_labeled_molecules(complete)
    out_adj, inc_adj, literal_out = build_molecule_indices(nt_path)             #Build the molecule indices

    # out_adj[u] = list of (predicate, object) for triples u --p--> o  (resource objects only)
    # Example (molecule d305 -> atom d305_24):
    #   out_adj["<http://dl-learner.org/carcinogenesis#d305>"] contains:
    #     ("<http://dl-learner.org/carcinogenesis#hasAtom>",
    #      "<http://dl-learner.org/carcinogenesis#d305_24>")
    
    # inc_adj[u] = list of (predicate, subject) for triples s --p--> u  (same edge, incoming view)
    # Example (atom d305_24 <- molecule d305):
    #   inc_adj["<http://dl-learner.org/carcinogenesis#d305_24>"] contains:
    #     ("<http://dl-learner.org/carcinogenesis#hasAtom>",
    #      "<http://dl-learner.org/carcinogenesis#d305>")
    
    # literal_out[u] = list of (predicate, literal) — object is not a URI, so no inc/out edge
    # Example (molecule d305 assay flag):
    #   literal_out["<http://dl-learner.org/carcinogenesis#d305>"] contains:
    #     ("<http://dl-learner.org/carcinogenesis#cytogen_ca>",
    #      '"false"^^<http://www.w3.org/2001/XMLSchema#boolean>')

    indices_dir = mutag_dir / "rdf_indices"
    out_csv, inc_csv, lit_csv = export_molecule_indices_to_csv(
        out_adj, inc_adj, literal_out, indices_dir
    )
    print(f"Wrote {out_csv.name}, {inc_csv.name}, {lit_csv.name} -> {indices_dir}")

    graphs: list[EgoGraph] = []
    for center, y in labels.items():
        graphs.append(ego_graph_for_molecule(center, y, out_adj, inc_adj, literal_out))   #Create ego graph for each molecule
    return graphs


def export_molecule_indices_to_csv(
    out_adj: dict[str, list[tuple[str, str]]],
    inc_adj: dict[str, list[tuple[str, str]]],
    literal_out: dict[str, list[tuple[str, str]]],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write out_adj, inc_adj, and literal_out as edge-list CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "out_adj.csv"
    inc_path = output_dir / "inc_adj.csv"
    lit_path = output_dir / "literal_out.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "object"])
        for subject, edges in out_adj.items():
            for predicate, obj in edges:
                w.writerow([subject, predicate, obj])

    with inc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object", "predicate", "subject"])
        for obj_node, edges in inc_adj.items():
            for predicate, subject in edges:
                w.writerow([obj_node, predicate, subject])

    with lit_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "literal"])
        for subject, edges in literal_out.items():
            for predicate, lit in edges:
                w.writerow([subject, predicate, lit])

    return out_path, inc_path, lit_path


def print_ego_graph_summary(mutag_dir: Path) -> None:
    """Build all ego graphs and print basic counts (for CLI / xai_mutag entry)."""
    graphs = build_all_ego_graphs(mutag_dir)
    empty = [g for g in graphs if not g.nodes]
    print(f"ego graphs: {len(graphs)} (labeled molecules)")
    print(f"empty (no match in .nt): {len(empty)}")
    sample = next(g for g in graphs if g.nodes)
    print(f"\nexample center: {sample.center}")
    print(f"  label: {sample.label}")
    print(f"  |nodes|: {len(sample.nodes)}")
    print(f"  |resource triples|: {len(sample.resource_triples)}")
    print(f"  |literal triples|: {len(sample.literal_triples)}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print_ego_graph_summary(root / "Datasets" / "mutag-hetero")

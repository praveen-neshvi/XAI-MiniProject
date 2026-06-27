

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer as _GNNExplainer
from torch_geometric.explain.config import ModelMode, ModelReturnType, ModelTaskLevel

from mutag_ego_graphs import RDF_TYPE
from mutag_rgcn_explainable import (
    DEFAULT_MODEL_PATH,
    load_explainable_model,
    prepare_molecule_data,
)

try:
    from visualize_ego_graph import short_label, short_predicate
    from visualize_explanation import write_explained_top_edges_html
except ImportError:
    def short_label(uri: str) -> str:
        if uri.startswith("<") and uri.endswith(">"):
            uri = uri[1:-1]
        return uri.split("#", 1)[-1] if "#" in uri else uri.rsplit("/", 1)[-1]

    def short_predicate(pred: str) -> str:
        frag = short_label(pred)
        return "rdf:type" if frag == "type" else frag

    write_explained_top_edges_html = None  # type: ignore[misc, assignment]


LABEL_NAMES = {0: "non-mutagenic", 1: "mutagenic"}


class GNNExplainerExcludeRoot(_GNNExplainer):
    """GNNExplainer with the prediction root node fixed (never feature-masked)."""

    def __init__(self, root_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.root_idx = root_idx

    def _train(
        self,
        model,
        x,
        edge_index,
        *,
        target,
        index=None,
        **kwargs,
    ):
        from torch_geometric.explain.algorithm.utils import set_masks

        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)
            loss.backward()
            optimizer.step()

            if self.node_mask is not None:
                with torch.no_grad():
                    self.node_mask.data[self.root_idx].fill_(20.0)

            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError(
                        "Could not compute gradients for node features."
                    )
                self.hard_node_mask = self.node_mask.grad != 0.0
                self.hard_node_mask[self.root_idx] = False
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x, edge_index):
        super()._initialize_masks(x, edge_index)
        if self.node_mask is not None:
            with torch.no_grad():
                self.node_mask.data[self.root_idx].fill_(20.0)


def relation_label(edge_type_id: int, pred2id: dict[str, int]) -> str:
    inv = {v: k for k, v in pred2id.items()}
    base_rel = edge_type_id // 2
    pred_uri = inv[base_rel]
    name = short_predicate(pred_uri)
    if edge_type_id % 2 == 1:
        return f"{name} (reverse)"
    return name


def node_rdf_type(node_uri: str, g) -> str:
    for s, p, o in g.resource_triples:
        if p == RDF_TYPE and s == node_uri:
            return short_label(o)
    return "unknown"


def node_importance_rows(g, data, node_mask: torch.Tensor) -> list[dict]:
    nodes = sorted(g.nodes)
    scores = node_mask.detach().cpu().view(-1)
    root_idx = int(data.root_idx.item())
    rows: list[dict] = []

    for node_idx, node_uri in enumerate(nodes):
        if node_idx == root_idx:
            continue
        rows.append(
            {
                "node_index": node_idx,
                "importance": float(scores[node_idx].item()),
                "node": node_uri,
                "node_label": short_label(node_uri),
                "rdf_type": node_rdf_type(node_uri, g),
                "is_root": node_idx == root_idx,
            }
        )
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def edge_importance_rows(
    g,
    data,
    edge_mask: torch.Tensor,
    pred2id: dict[str, int],
) -> list[dict]:
    nodes = sorted(g.nodes)
    scores = edge_mask.detach().cpu().view(-1)
    rows: list[dict] = []
    edge_index = data.edge_index.cpu()
    edge_type = data.edge_type.cpu()

    for e_idx in range(edge_index.size(1)):
        src = int(edge_index[0, e_idx])
        dst = int(edge_index[1, e_idx])
        rows.append(
            {
                "edge_index": e_idx,
                "importance": float(scores[e_idx].item()),
                "subject": nodes[src],
                "object": nodes[dst],
                "predicate": relation_label(int(edge_type[e_idx]), pred2id),
                "subject_label": short_label(nodes[src]),
                "object_label": short_label(nodes[dst]),
                "edge_type_id": int(edge_type[e_idx]),
            }
        )
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def write_node_explanation_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "importance",
        "node_label",
        "rdf_type",
        "is_root",
        "node",
        "node_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "importance": f"{row['importance']:.6f}",
                    "node_label": row["node_label"],
                    "rdf_type": row["rdf_type"],
                    "is_root": row["is_root"],
                    "node": row["node"],
                    "node_index": row["node_index"],
                }
            )


def write_explanation_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "importance",
        "subject_label",
        "predicate",
        "object_label",
        "subject",
        "object",
        "edge_type_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "importance": f"{row['importance']:.6f}",
                    "subject_label": row["subject_label"],
                    "predicate": row["predicate"],
                    "object_label": row["object_label"],
                    "subject": row["subject"],
                    "object": row["object"],
                    "edge_type_id": row["edge_type_id"],
                }
            )


def explain_molecule(
    molecule_id: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    mutag_dir: Path | None = None,
    explainer_epochs: int = 200,
    top_k: int = 10,
    explain_class: str = "prediction",
) -> dict:
    if mutag_dir is None:
        mutag_dir = Path(__file__).resolve().parent / "Datasets" / "mutag-hetero"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = load_explainable_model(model_path, device=device)
    g, data, uri = prepare_molecule_data(molecule_id, bundle, mutag_dir)
    data = data.to(device)

    root_idx = int(data.root_idx.item())
    with torch.no_grad():
        node_logits = model(data.x, data.edge_index, data.edge_type)
        root_logit = node_logits[root_idx]
        probs = F.softmax(root_logit, dim=-1)
        pred_class = int(root_logit.argmax().item())

    true_label = int(g.label)
    if explain_class == "label":
        target_class = true_label
        explanation_type = "phenomenon"
    else:
        target_class = pred_class
        explanation_type = "model"

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainerExcludeRoot(root_idx=root_idx, epochs=explainer_epochs),
        explanation_type=explanation_type,
        node_mask_type="object",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode=ModelMode.multiclass_classification,
            task_level=ModelTaskLevel.node,
            return_type=ModelReturnType.raw,
        ),
        threshold_config=dict(threshold_type="topk", value=top_k),
    )

    explain_kwargs = dict(
        x=data.x,
        edge_index=data.edge_index,
        edge_type=data.edge_type,
        index=root_idx,
    )
    if explanation_type == "phenomenon":
        explain_kwargs["target"] = target_class

    explanation = explainer(**explain_kwargs)

    edge_mask = explanation.edge_mask
    if edge_mask is None:
        raise RuntimeError("GNNExplainer did not return an edge_mask.")

    ranked_edges = edge_importance_rows(g, data, edge_mask, bundle["pred2id"])

    node_mask = explanation.node_mask
    if node_mask is None:
        raise RuntimeError("GNNExplainer did not return a node_mask.")
    ranked_nodes = node_importance_rows(g, data, node_mask)

    return {
        "molecule_id": molecule_id,
        "uri": uri,
        "ego_graph": g,
        "true_label": true_label,
        "pred_class": pred_class,
        "probs": probs.cpu().tolist(),
        "root_idx": root_idx,
        "target_class": target_class,
        "ranked_edges": ranked_edges,
        "top_edges": ranked_edges[:top_k],
        "edge_mask": edge_mask.cpu(),
        "ranked_nodes": ranked_nodes,
        "top_nodes": ranked_nodes[:top_k],
        "node_mask": node_mask.cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain MUTAG predictions with GNNExplainer (node-level FastRGCN).",
    )
    parser.add_argument("molecule", help="Molecule id, e.g. d305")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to models/rgcn_explainable.pt",
    )
    parser.add_argument("--mutag-dir", type=Path, default=None)
    parser.add_argument("--explainer-epochs", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--explain-class",
        choices=("prediction", "label"),
        default="prediction",
        help="Explain predicted class or true label",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Edge CSV path (default: visualizations/explain_<id>.csv)",
    )
    parser.add_argument(
        "--nodes-csv-out",
        type=Path,
        default=None,
        help="Node CSV path (default: visualizations/explain_<id>_nodes.csv)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip explained subgraph HTML",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Explained graph HTML (default: visualizations/explain_<id>_graph.html)",
    )
    args = parser.parse_args()

    result = explain_molecule(
        args.molecule,
        model_path=args.model,
        mutag_dir=args.mutag_dir,
        explainer_epochs=args.explainer_epochs,
        top_k=args.top_k,
        explain_class=args.explain_class,
    )

    print(f"model: {args.model}")
    print(f"molecule: {result['molecule_id']}")
    print(f"uri: {result['uri']}")
    print(f"root node index: {result['root_idx']}")
    print()
    print("--- Prediction (root node) ---")
    print(f"true label:      {result['true_label']} ({LABEL_NAMES[result['true_label']]})")
    print(f"predicted class: {result['pred_class']} ({LABEL_NAMES[result['pred_class']]})")
    print(
        "probabilities:   "
        f"P(non-mutagenic)={result['probs'][0]:.4f}, "
        f"P(mutagenic)={result['probs'][1]:.4f}"
    )
    print()
    print("--- GNNExplainer (node + edge masks) ---")
    print(
        f"explained target class: {result['target_class']} "
        f"({LABEL_NAMES[result['target_class']]})"
    )
    print(f"top-{len(result['top_nodes'])} nodes (root excluded from node mask):")
    for rank, row in enumerate(result["top_nodes"], start=1):
        print(
            f"  {rank:2d}. score={row['importance']:.4f}  "
            f"{row['node_label']} ({row['rdf_type']})"
        )
    print()
    print(f"top-{len(result['top_edges'])} edges:")
    for rank, row in enumerate(result["top_edges"], start=1):
        print(
            f"  {rank:2d}. score={row['importance']:.4f}  "
            f"{row['subject_label']} --{row['predicate']}--> {row['object_label']}"
        )

    viz_dir = Path(__file__).resolve().parent / "visualizations"
    edge_csv_path = args.csv_out or viz_dir / f"explain_{args.molecule}.csv"
    node_csv_path = args.nodes_csv_out or viz_dir / f"explain_{args.molecule}_nodes.csv"
    write_explanation_csv(result["ranked_edges"], edge_csv_path)
    write_node_explanation_csv(result["ranked_nodes"], node_csv_path)
    print(f"\nWrote ranked edges to {edge_csv_path}")
    print(f"Wrote ranked nodes to {node_csv_path}")

    if not args.no_html and write_explained_top_edges_html is not None:
        html_path = args.html_out or viz_dir / f"explain_{args.molecule}_graph.html"
        write_explained_top_edges_html(
            result["ego_graph"],
            result["ranked_nodes"],
            result["ranked_edges"],
            html_path,
            top_k=args.top_k,
            target_class=result["target_class"],
            label_names=LABEL_NAMES,
        )
        print(f"Wrote explained graph HTML to {html_path}")


if __name__ == "__main__":
    main()

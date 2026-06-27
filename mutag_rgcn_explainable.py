"""
MUTAG node-classification R-GCN for GNNExplainer (Strategy 1 style).

Pipeline per molecule ego graph:
  one-hot node features (x) -> FastRGCNConv -> FastRGCNConv -> root node logits

Differences from mutag_rgcn.py (graph classification):
  - Bidirectional edges (forward + reverse relations, like AIFB)
  - FastRGCNConv instead of RGCNConv + embeddings
  - Loss only on the root (molecule center) node
  - No global_mean_pool or graph_lit_feat

Examples:
  python mutag_rgcn_explainable.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import FastRGCNConv

from mutag_ego_graphs import (
    RDF_TYPE,
    build_molecule_indices,
    ego_graph_for_molecule,
    load_labeled_molecules,
    uri_to_nt,
)
from mutag_rgcn import (
    _vocab_from_graphs,
    accuracy,
    confusion_matrix,
    load_split_molecules,
    macro_f1,
    per_class_prf,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "rgcn_explainable.pt"


def ego_graph_to_node_data(
    g,
    pred2id: dict[str, int],
    type2id: dict[str, int],
    unk_type_id: int = 0,
) -> Data:
    """PyG Data with one-hot x, bidirectional edges, label on root node."""
    nodes = sorted(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    root_idx = idx[g.center]
    num_types = len(type2id) + 1

    x = torch.zeros((len(nodes), num_types), dtype=torch.float32)
    for s, p, o in g.resource_triples:
        if p == RDF_TYPE and s in idx:
            tid = type2id.get(o, unk_type_id)
            x[idx[s], tid] = 1.0
    for i in range(len(nodes)):
        if x[i].sum() == 0:
            x[i, unk_type_id] = 1.0

    ei: list[list[int]] = []
    et: list[int] = []
    for s, p, o in g.resource_triples:
        rel = pred2id[p]
        src, dst = idx[s], idx[o]
        ei.append([src, dst])
        et.append(2 * rel)
        ei.append([dst, src])
        et.append(2 * rel + 1)

    if ei:
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(et, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    y = torch.tensor([g.label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    data.root_idx = torch.tensor([root_idx], dtype=torch.long)
    data.num_nodes = len(nodes)
    return data


def collate_graphs(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


def root_logits(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Select per-graph logits at each molecule's root node."""
    roots = batch.ptr[:-1] + batch.root_idx.view(-1)
    return logits[roots]


class FastRGCNRootClassifier(nn.Module):
    """Node-level FastRGCN; molecule label is predicted at the root node."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_relations: int,
        num_classes: int = 2,
        num_bases: int = 8,
    ):
        super().__init__()
        bases = min(num_bases, num_relations) if num_relations else 1
        bases = max(bases, 1)
        self.conv1 = FastRGCNConv(in_channels, hidden, num_relations, num_bases=bases)
        self.conv2 = FastRGCNConv(hidden, num_classes, num_relations, num_bases=bases)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_type).relu()
        return self.conv2(x, edge_index, edge_type)


def save_explainable_model(
    path: Path,
    model: FastRGCNRootClassifier,
    *,
    pred2id: dict[str, int],
    type2id: dict[str, int],
    in_channels: int,
    num_relations: int,
    hyperparams: dict[str, Any],
    best_epoch: int,
    best_val_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "pred2id": pred2id,
            "type2id": type2id,
            "in_channels": in_channels,
            "num_relations": num_relations,
            "hyperparams": hyperparams,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "task": "node_classification",
            "bidirectional": True,
        },
        path,
    )


def load_explainable_model(
    path: Path,
    device: torch.device | None = None,
) -> tuple[FastRGCNRootClassifier, dict[str, Any]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt["hyperparams"]
    model = FastRGCNRootClassifier(
        ckpt["in_channels"],
        hp["hidden"],
        ckpt["num_relations"],
        num_classes=hp.get("num_classes", 2),
        num_bases=hp["num_bases"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bundle = {
        "pred2id": ckpt["pred2id"],
        "type2id": ckpt["type2id"],
        "in_channels": ckpt["in_channels"],
        "num_relations": ckpt["num_relations"],
        "hyperparams": hp,
        "best_epoch": ckpt["best_epoch"],
        "best_val_f1": ckpt["best_val_f1"],
        "task": ckpt.get("task", "node_classification"),
        "bidirectional": ckpt.get("bidirectional", True),
    }
    return model, bundle


def resolve_molecule_uri(molecule_id: str, labels: dict[str, int]) -> str:
    candidate = molecule_id.strip()
    if candidate in labels:
        return candidate
    if not candidate.startswith("<"):
        candidate = uri_to_nt(f"http://dl-learner.org/carcinogenesis#{candidate}")
    if candidate in labels:
        return candidate
    bare = uri_to_nt(candidate)
    if bare in labels:
        return bare
    raise ValueError(f"Unknown molecule {molecule_id!r} (not in completeDataset.tsv)")


def prepare_molecule_data(
    molecule_id: str,
    bundle: dict[str, Any],
    mutag_dir: Path,
):
    complete = mutag_dir / "completeDataset.tsv"
    nt_path = mutag_dir / "mutag_stripped.nt"
    labels = load_labeled_molecules(complete)
    uri = resolve_molecule_uri(molecule_id, labels)
    out_adj, inc_adj, literal_out = build_molecule_indices(nt_path)
    g = ego_graph_for_molecule(uri, labels[uri], out_adj, inc_adj, literal_out)
    data = ego_graph_to_node_data(g, bundle["pred2id"], bundle["type2id"])
    return g, data, uri


def run_explainable_training(
    mutag_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    num_bases: int = 8,
    val_ratio: float = 0.2,
    model_path: Path | None = DEFAULT_MODEL_PATH,
) -> None:
    complete = mutag_dir / "completeDataset.tsv"
    nt_path = mutag_dir / "mutag_stripped.nt"
    train_tsv = mutag_dir / "trainingSet.tsv"
    test_tsv = mutag_dir / "testSet.tsv"

    labels = load_labeled_molecules(complete)
    train_uris, test_uris = load_split_molecules(train_tsv, test_tsv)
    out_adj, inc_adj, literal_out = build_molecule_indices(nt_path)

    train_graphs = [
        ego_graph_for_molecule(u, labels[u], out_adj, inc_adj, literal_out) for u in train_uris
    ]
    test_graphs = [
        ego_graph_for_molecule(u, labels[u], out_adj, inc_adj, literal_out) for u in test_uris
    ]

    pred2id, type2id = _vocab_from_graphs(train_graphs + test_graphs)
    num_relations = 2 * len(pred2id)
    in_channels = len(type2id) + 1

    train_data_all = [ego_graph_to_node_data(g, pred2id, type2id) for g in train_graphs]
    test_data = [ego_graph_to_node_data(g, pred2id, type2id) for g in test_graphs]

    split = max(1, int(len(train_data_all) * (1.0 - val_ratio)))
    split = min(split, len(train_data_all) - 1)
    train_data = train_data_all[:split]
    val_data = train_data_all[split:]

    train_labels = torch.cat([d.y for d in train_data], dim=0)
    cls_counts = torch.bincount(train_labels, minlength=2).float()
    class_weight = cls_counts.sum() / (2.0 * cls_counts.clamp_min(1.0))

    hyperparams = {
        "hidden": hidden,
        "num_bases": num_bases,
        "num_classes": 2,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastRGCNRootClassifier(
        in_channels,
        hidden,
        num_relations,
        num_classes=2,
        num_bases=num_bases,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weight = class_weight.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

    print(f"device: {device}")
    print(f"task: node classification (root node)")
    print(f"train molecules: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    print(f"num_relations (bidirectional): {num_relations}, in_channels (one-hot types): {in_channels}")
    print(
        f"class weights (for CE): 0->{class_weight[0].item():.3f}, 1->{class_weight[1].item():.3f}"
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            node_logits = model(batch.x, batch.edge_index, batch.edge_type)
            logits = root_logits(node_logits, batch)
            loss = F.cross_entropy(logits, batch.y, weight=class_weight)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * batch.num_graphs
        train_loss = total_loss / len(train_data)

        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                node_logits = model(batch.x, batch.edge_index, batch.edge_type)
                all_logits.append(root_logits(node_logits, batch).cpu())
                all_y.append(batch.y.cpu())
        val_logits = torch.cat(all_logits, dim=0)
        val_y = torch.cat(all_y, dim=0)
        val_acc = accuracy(val_logits, val_y)
        val_f1 = macro_f1(val_logits, val_y)
        val_cm = confusion_matrix(val_logits, val_y)
        val_prf = per_class_prf(val_logits, val_y)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}"
        )
        print(
            "  val_confusion [[TN, FP],[FN, TP]] = "
            f"[[{val_cm[0,0].item()}, {val_cm[0,1].item()}], "
            f"[{val_cm[1,0].item()}, {val_cm[1,1].item()}]]"
        )
        print(
            "  class_0 precision/recall/f1 = "
            f"{val_prf[0][0]:.3f}/{val_prf[0][1]:.3f}/{val_prf[0][2]:.3f}"
        )
        print(
            "  class_1 precision/recall/f1 = "
            f"{val_prf[1][0]:.3f}/{val_prf[1][1]:.3f}/{val_prf[1][2]:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            node_logits = model(batch.x, batch.edge_index, batch.edge_type)
            all_logits.append(root_logits(node_logits, batch).cpu())
            all_y.append(batch.y.cpu())

    te_logits = torch.cat(all_logits, dim=0)
    te_y = torch.cat(all_y, dim=0)
    te_acc = accuracy(te_logits, te_y)
    te_f1 = macro_f1(te_logits, te_y)
    te_cm = confusion_matrix(te_logits, te_y)
    te_prf = per_class_prf(te_logits, te_y)
    print("\n--- Final test (best val checkpoint) ---")
    print(f"best_epoch={best_epoch}  best_val_macro_f1={best_val_f1:.4f}")
    print(f"test_acc={te_acc:.4f}  test_macro_f1={te_f1:.4f}")
    print(
        "test_confusion [[TN, FP],[FN, TP]] = "
        f"[[{te_cm[0,0].item()}, {te_cm[0,1].item()}], "
        f"[{te_cm[1,0].item()}, {te_cm[1,1].item()}]]"
    )
    print(
        "test class_0 precision/recall/f1 = "
        f"{te_prf[0][0]:.3f}/{te_prf[0][1]:.3f}/{te_prf[0][2]:.3f}"
    )
    print(
        "test class_1 precision/recall/f1 = "
        f"{te_prf[1][0]:.3f}/{te_prf[1][1]:.3f}/{te_prf[1][2]:.3f}"
    )

    if model_path is not None and best_state is not None:
        save_explainable_model(
            model_path,
            model,
            pred2id=pred2id,
            type2id=type2id,
            in_channels=in_channels,
            num_relations=num_relations,
            hyperparams=hyperparams,
            best_epoch=best_epoch,
            best_val_f1=best_val_f1,
        )
        print(f"\nSaved best model to {model_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "Datasets" / "mutag-hetero"
    run_explainable_training(root, epochs=50)

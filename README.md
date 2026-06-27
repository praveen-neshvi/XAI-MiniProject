# XAI Mini Project — MUTAG Mutagenicity with R-GCN and GNNExplainer

Binary classification of MUTAG compounds (mutagenic / non-mutagenic) from **RDF ego graphs**, using a **FastR-GCN** node classifier and **GNNExplainer** for post-hoc explanations (Strategy 1 style).

---

## Repository structure

```
XAI-MiniProject/
├── README.md
├── requirements.txt
│
├── mutag_ego_graphs.py          # Build molecule-centered ego graphs from RDF
├── mutag_rgcn_explainable.py    # Train FastRGCN (node classification at molecule center)
├── explain_rgcn.py              # GNNExplainer: ranked nodes & edges + CSV export
├── visualize_ego_graph.py       # Full ego graph → interactive HTML (PyVis)
├── visualize_explanation.py     # Explained ego graph → interactive HTML (PyVis)
│
└── Datasets/
    └── mutag-hetero/
        ├── completeDataset.tsv  # All labeled molecules (URI + label)
        ├── mutag_stripped.nt    # RDF triples (atoms, bonds, types, …)
        ├── trainingSet.tsv      # Train split (molecule URIs)
        └── testSet.tsv          # Test split (molecule URIs)
```

### Generated at runtime (not committed)

```
models/
└── rgcn_explainable.pt          # Best checkpoint after training

visualizations/
├── ego_<id>.html                # Full ego graph (from visualize_ego_graph.py)
├── explain_<id>.csv             # All edges ranked by importance
├── explain_<id>_nodes.csv       # All nodes ranked by importance
└── explain_<id>_graph.html      # Full ego graph with top-k highlights

Datasets/mutag-hetero/rdf_indices/   # Optional CSV exports from ego-graph indexing
```

---

## What each file does

| File | Purpose |
|------|---------|
| **`mutag_ego_graphs.py`** | Reads `mutag_stripped.nt` and labels; builds one **ego graph per molecule** (BFS from compound center). Skips expanding through shared `rdf:type` class hubs so graphs stay local. |
| **`mutag_rgcn_explainable.py`** | Converts ego graphs to PyG data (one-hot `rdf:type`, bidirectional edges), trains **2-layer FastRGCN**, predicts mutagenicity at the **root (molecule) node**. Saves `models/rgcn_explainable.pt`. |
| **`explain_rgcn.py`** | Loads the trained model, runs **GNNExplainer** on one molecule: top-k **nodes** and **edges**, console summary, CSVs. Root node is **not** node-masked (fixed as prediction target). Optionally writes explained HTML. |
| **`visualize_ego_graph.py`** | PyVis HTML for the **full** ego graph of one molecule (all nodes/edges). |
| **`visualize_explanation.py`** | PyVis HTML for the **full** ego graph with **top-k nodes/edges highlighted** and the rest grayed out; center molecule in **red**. |
| **`requirements.txt`** | Python dependencies (`torch`, `torch-geometric`, `pyvis`). |

### Dataset files

| File | Content |
|------|---------|
| **`completeDataset.tsv`** | Every labeled compound: molecule URI + `label_mutagenic` (0/1). |
| **`mutag_stripped.nt`** | RDF graph: triples linking compounds, atoms, bonds, ring classes, etc. |
| **`trainingSet.tsv`** | Molecule URIs for training (validation split is taken from train in code). |
| **`testSet.tsv`** | Molecule URIs for testing. |

---

## Setup

```bash
cd XAI-MiniProject

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For GPU/CUDA, install the matching PyTorch wheel from [pytorch.org](https://pytorch.org/) first, then run `pip install -r requirements.txt` again.

---

## How to run (in order)

### 1. Train the explainable model

```bash
python mutag_rgcn_explainable.py
```

- Input: `Datasets/mutag-hetero/`
- Output: `models/rgcn_explainable.pt`
- Task: node classification at the molecule center (50 epochs by default)

### 2. Explain one molecule (GNNExplainer)

```bash
python explain_rgcn.py d305
```

Options:

```bash
python explain_rgcn.py d305 --top-k 10 --explainer-epochs 200
python explain_rgcn.py d187 --explain-class label    # explain true label, not prediction
python explain_rgcn.py d305 --no-html                # skip HTML graph
```

Outputs:

- Console: prediction + top-k nodes/edges
- `visualizations/explain_d305.csv`
- `visualizations/explain_d305_nodes.csv`
- `visualizations/explain_d305_graph.html` (unless `--no-html`)

### 3. Visualize full ego graph (optional)

```bash
python visualize_ego_graph.py d305
python visualize_ego_graph.py d305 --open
python visualize_ego_graph.py --list
```

Output: `visualizations/ego_d305.html`

### 4. Visualize explained graph only (optional)

```bash
python visualize_explanation.py d305 --top-k 10
```

Output: `visualizations/explain_d305_graph.html`  
(Runs explainer + HTML in one step; same result as `explain_rgcn.py` without `--no-html`.)

---

## Pipeline overview

```
RDF (.nt) + labels (.tsv)
        ↓
mutag_ego_graphs.py          →  one ego graph per molecule
        ↓
mutag_rgcn_explainable.py    →  FastRGCN, predict at molecule node
        ↓
explain_rgcn.py              →  GNNExplainer (node + edge masks)
        ↓
visualize_explanation.py     →  HTML: top-k highlighted, rest gray
```

---

## Model summary (for reports)

- **Input:** Ego graph per molecule; node features = one-hot `rdf:type`; edges = typed RDF predicates (bidirectional).
- **Model:** 2 × `FastRGCNConv` (hidden 64), logits at **root node** only.
- **Explanation:** PyG `GNNExplainer`, `task_level=node`, `index=root_idx`; root excluded from node masking; top-k nodes and edges ranked by importance.

---

## Example molecule IDs

Use IDs from the dataset, e.g. `d305`, `d187`, `d42`. List labeled compounds:

```bash
python visualize_ego_graph.py --list
```

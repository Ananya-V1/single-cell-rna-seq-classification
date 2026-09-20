# Single-Cell RNA-seq Classification

Clusters single-cell RNA-seq data by cell type (unsupervised) and trains classifiers to
predict cluster identity from gene expression, using the resulting feature importances
and marker genes to interpret and label the clusters.

## Dataset

[`pbmc3k`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html) —
~2,700 peripheral blood mononuclear cells (PBMCs), a standard single-cell RNA-seq benchmark
dataset. Downloaded automatically by `scanpy` on first run; no manual download needed.

## Pipeline (`rna.py`)

1. **Load & preprocess** — normalize per-cell counts, log-transform, subset to the 2,000
   most variable genes, and scale (z-score) for downstream analysis.
2. **Cluster** — PCA → k-nearest-neighbor graph → Leiden community detection to group cells
   by transcriptional similarity (an unsupervised proxy for cell type). Clusters with fewer
   than 50 cells are dropped.
3. **Classify** — train Random Forest and Gradient Boosting models to predict Leiden
   cluster from gene expression, and inspect feature importances (which genes drive each
   the model's predictions).
4. **Annotate** — run a Wilcoxon rank-sum test per cluster to find marker genes, then map
   cluster IDs to known PBMC cell types (CD4/CD8 T cells, B cells, NK cells, monocytes) based
   on the identified markers.
5. **Visualize** — project cells into 2D with UMAP, colored by both raw cluster ID and
   annotated cell type.

## Setup

Requires Python 3.9+.

```bash
pip install pandas matplotlib scanpy scikit-learn
```

## Run

```bash
python3 rna.py
```

## Output

- Console: dataset shape/metadata, per-cluster cell counts, classifier accuracy/CV scores
  and classification reports, and top-10 marker genes per cluster.
- `FeatureImportance_rf.png` / `FeatureImportance_gb.png` — top 20 genes by importance for
  each classifier.
- `figures/umap_clusters.png` — UMAP plot colored by cluster and by annotated cell type
  (generated locally; not committed, since `figures/` is gitignored).

## Notes

- Cell-type labels in `celltype` (in `rna.py`) were assigned manually by matching each
  cluster's marker genes to known PBMC biology — re-run and re-check this mapping if the
  clustering output changes (e.g., different cluster count or composition).

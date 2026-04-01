import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

adata = sc.datasets.pbmc3k()

print(adata.X.shape)
print(adata.obs.head())
print(adata.var.head())

# Scale cells so total gene count sums to 10000
sc.pp.normalize_total(adata, target_sum=1e4)
# Compress range
sc.pp.log1p(adata)

# Keeps 2000 most variable genes; remove other genes to reduce noise
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()

# Saves the values
adata.layers['log_norm'] = adata.X.copy()
# Center genes to mean = 0, std = 1
sc.pp.scale(adata)


sc.tl.pca(adata, n_comps=40)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
# Split graph into clusters of similar cells (corresponds with cell type) based on gene expression
sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)

cluster_counts = adata.obs['leiden'].value_counts()

# Removes clusters with < 50; not enough information to train
keep = cluster_counts[cluster_counts >= 50].index
adata = adata[adata.obs['leiden'].isin(keep)].copy()

X = adata.X
y = adata.obs['leiden'] # leiden clusters based on cell type

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predict cell type based on cell's gene expression
# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_rf_pred) # calculate accuracy score
print("Random Forest Classifier val_accuracy: %.4f" % rf_acc)
print(classification_report(y_test, y_rf_pred, zero_division=0))

rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("Random Forest CV accuracy: %.4f +/- %.4f" % (rf_cv.mean(), rf_cv.std()))

# Random Forest - Feature Importance
importances_rf = pd.Series(rf.feature_importances_, index=adata.var_names)
importances_rf.nlargest(20).sort_values().plot(kind='barh', figsize=(8, 6), color='steelblue')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/Users/ananya/rna_seq/FeatureImportance_rf.png", dpi=300)
plt.close()

# Gradient Boosting
gb = HistGradientBoostingClassifier(n_iter_no_change=10, random_state=42, class_weight='balanced')
gb.fit(X_train, y_train)
y_gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, y_gb_pred)
print("Gradient Boosting Classifier val_accuracy: %.4f" % gb_acc)
print(classification_report(y_test, y_gb_pred, zero_division=0))

gb_cv = cross_val_score(gb, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("Gradient Boosting CV accuracy: %.4f +/- %.4f" % (gb_cv.mean(), gb_cv.std()))

# Gradient Boosting - Feature Importance
perm = permutation_importance(gb, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
importances_gb = pd.Series(perm.importances_mean, index=adata.var_names)
importances_gb.nlargest(20).sort_values().plot(kind='barh', figsize=(8, 6), color='darkorange')
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/Users/ananya/rna_seq/FeatureImportance_gb.png", dpi=300)
plt.close()


# Marker genes
adata.X = adata.layers['log_norm']
sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')

for cluster in adata.obs['leiden'].unique():
    genes = adata.uns['rank_genes_groups']['names'][cluster][:10]
    print(f"Cluster {cluster}: {genes}")

# Cell type annotation -> maps cluster IDS to cell type names
celltype = {
    '0': 'CD4 T cells',
    '1': 'CD4 T cells',
    '2': 'B cells',
    '3': 'CD14+ Monocytes',
    '4': 'CD16+ Monocytes',
    '5': 'NK cells',
    '7': 'CD8+ T cells'
}
adata.obs['cell_type'] = adata.obs['leiden'].map(celltype)

# UMAP -> projects cells into 2D
sc.tl.umap(adata)
sc.pl.umap(adata, color=['leiden', 'cell_type'], save='_clusters.png')

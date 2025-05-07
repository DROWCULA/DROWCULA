# DROWCULA: Dimensionally Reduced Open‑World Clustering

Working with annotated data is the cornerstone of supervised learning—but labeling is costly, and in open‑world settings new categories may emerge at any time. **DROWCULA** (Dimensionally Reduced Open‑World Clustering) offers a **fully unsupervised** solution for discovering both the number of novel classes and their assignments in image datasets. By combining Vision‑Transformer embeddings with manifold‑learning refinement, DROWCULA sets new state‑of‑the‑art in single‑modal clustering and Novel Class Discovery on CIFAR‑10, CIFAR‑100, ImageNet‑100, and Tiny ImageNet, both when the cluster count is known and when it must be estimated.


## Results for DROWCULA UMAP

| **Dataset**       | **Known‑K ACC (%)** | **Unknown‑K ACC (%)** |
|-------------------|--------------------:|----------------------:|
| CIFAR‑10          | 99.1                | 95.4                  |
| CIFAR‑100         | 80.4                | 80.0                  |
| ImageNet‑100      | 83.5                | 83.8                  |
| Tiny ImageNet     | 77.5                | 78.4                  |


## Results for DROWCULA t-SNE

| **Dataset**       | **Known‑K ACC (%)** | **Unknown‑K ACC (%)** |
|-------------------|--------------------:|----------------------:|
| CIFAR‑10          | 98.6                | 90.3                  |
| CIFAR‑100         | 81.8                | 79.0                  |
| ImageNet‑100      | 88.6                | 85.2                  |
| Tiny ImageNet     | 78.4                | 79.4                  |

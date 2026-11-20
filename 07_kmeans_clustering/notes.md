# 07. K-Means Clustering (Part 1 & 2)

## Part 1: Finding Groups Without Labels
Now we drop the answer key completely. Imagine opening $K = 4$ delivery hubs in a brand-new city so that every house is as close as possible to its nearest hub. Where should you place the hubs? That is **K-Means Clustering**.

### The 4-Step Algorithm
1. **Initialize:** Place $K$ starting center points (**centroids**) on the map.
2. **Assignment Step:** Every data point joins whichever centroid is closest (Euclidean distance).
3. **Update Step:** Move each centroid to the exact mean (center of mass) of all points assigned to it.
4. **Repeat:** Keep alternating Assignment and Update until the centroids stop moving.

## Part 2: Picking $K$, Initialization & Limitations
### The Elbow Method (WCSS / Inertia)
We measure cluster compactness using **WCSS (Within-Cluster Sum of Squares)**:
$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$
- Adding more clusters always lowers WCSS (at $K = N$, every point is its own cluster and WCSS $= 0$, which tells you nothing).
- Plot WCSS against $K$: it plunges sharply at first, then flattens out like a bent arm. The sharp bend—the **elbow**—indicates the natural number of clusters.

### Watch-Outs Every Practitioner Should Know
- **Scale your features first:** Just like KNN, K-Means relies on Euclidean distance.
- **Random initialization trap:** Bad starting centroids can get stuck in a suboptimal local minimum. Running multiple restarts or using **K-Means++** fixes this.
- **Shape assumption:** K-Means assumes roughly spherical, similar-sized blobs. For curved crescents or rings, density-based methods like DBSCAN work better.

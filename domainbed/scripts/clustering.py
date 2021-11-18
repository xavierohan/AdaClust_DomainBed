import faiss


class Faiss_Clustering:
    def __init__(self, X, n_clusters, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.X = X
        self.kmeans = faiss.Kmeans(
            d=self.X.shape[1], k=self.n_clusters, niter=self.max_iter, nredo=self.n_init, gpu=True
        )

    def fit(self):
        self.kmeans.train(self.X)
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
        self.stats = self.kmeans.iteration_stats[-1]

    def predict(self, X):
        return self.kmeans.index.search(X, 1)[1]

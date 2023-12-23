from scipy.linalg import eigh


class PCA:
    def __init__(self, n_components, eps=1e-8):
        self.n_components = n_components
        self.eps = eps
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X_standard = self._standardize(X)
        
        n_samples, n_features = X.shape
        covariane_matrix = X_standard.T @ X_standard / (n_samples - 1)
        
        eig_vals, eig_vects = eigh(
            covariane_matrix,
            subset_by_index=(n_features-self.n_components, n_features-1)    # top `n_components`
        )
        
        self.eig_vals, self.eig_vects = eig_vals[::-1], eig_vects[::-1]   # to descending order

        return self
    
    def transform(self, X):
        X_standard = self._standardize(X)
        X_reduced = X_standard @ self.eig_vects
        return X_reduced
    
    def auto_encode(self, X):
        X_standard = self._standardize(X)
        X_autoencoded = X_standard @ self.eig_vects @ self.eig_vects.T * self.std + self.mean
        return X_autoencoded
    
    def _standardize(self, X):
        return (X - self.mean) / (self.std + self.eps)

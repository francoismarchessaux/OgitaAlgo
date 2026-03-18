from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def _as_2d_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    return arr


def _as_column_vector(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1, 1)
    return arr


def sorted_eigh(A: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of a real symmetric matrix, sorted descending."""
    A = np.asarray(A, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def estimate_eigenvalues(
    A: ArrayLike,
    X_hat: ArrayLike,
    return_extra: bool = False,
):
    """
    Paper Algorithm 1.

    Parameters
    ----------
    A : (p, p) real symmetric matrix
        Covariance matrix or other symmetric matrix.
    X_hat : (p, p) matrix
        Approximate eigenvectors in columns.
    return_extra : bool
        If True, also return R and S from the paper.
    """
    A = np.asarray(A, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if X_hat.shape != A.shape:
        raise ValueError("X_hat must have the same shape as A.")

    n = A.shape[0]
    R = np.eye(n) - X_hat.T @ X_hat
    S = X_hat.T @ A @ X_hat

    denom = 1.0 - np.diag(R)
    tiny = 1e-15
    denom = np.where(np.abs(denom) < tiny, tiny, denom)
    lambdas = np.diag(S) / denom

    if return_extra:
        return {"lambda": lambdas, "R": R, "S": S}
    return lambdas


def ogita_aishima_step(A: ArrayLike, X_hat: ArrayLike) -> np.ndarray:
    """Paper Algorithm 2: one Ogita-Aishima refinement step."""
    extra = estimate_eigenvalues(A=A, X_hat=X_hat, return_extra=True)
    lambdas = extra["lambda"]
    R = extra["R"]
    S = extra["S"]

    D = np.diag(lambdas)
    delta = 2.0 * (
        np.linalg.norm(S - D, ord=2)
        + np.linalg.norm(A, ord=2) * np.linalg.norm(R, ord=2)
    )

    n = A.shape[0]
    E = np.empty((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if abs(lambdas[i] - lambdas[j]) > delta:
                E[i, j] = (S[i, j] + lambdas[j] * R[i, j]) / (lambdas[j] - lambdas[i])
            else:
                E[i, j] = 0.5 * R[i, j]

    return np.asarray(X_hat, dtype=float) + np.asarray(X_hat, dtype=float) @ E


def ogita_aishima(
    A: ArrayLike,
    X_hat: ArrayLike,
    tol: float = 1e-6,
    max_iter_count: int | None = None,
    sort_by_eigenvalues: bool = False,
    return_extra: bool = False,
):
    """
    Paper Algorithm 3: iterate refinement until convergence.
    """
    if tol <= 0:
        raise ValueError("tol must be strictly positive.")

    current = np.asarray(X_hat, dtype=float)
    iter_count = 0
    epsilon = np.inf

    while True:
        iter_count += 1
        updated = ogita_aishima_step(A=A, X_hat=current)
        epsilon = np.linalg.norm(updated - current, ord=2)
        current = updated

        if max_iter_count is not None and iter_count >= max_iter_count:
            break
        if epsilon < tol:
            break

    lambdas = None
    if sort_by_eigenvalues:
        lambdas = estimate_eigenvalues(A=A, X_hat=current, return_extra=False)
        order = np.argsort(lambdas)[::-1]
        lambdas = lambdas[order]
        current = current[:, order]

    if return_extra:
        return {
            "result": current,
            "lambda": lambdas,
            "iter_count": iter_count,
            "epsilon": epsilon,
        }
    return current


class IPCA:
    """
    Iterated PCA from the paper.

    Successive calls to fit() reuse the previous eigenvectors as the initial guess
    for the next covariance matrix.
    """

    def __init__(self, tol: float = 1e-6, max_iter_count: int | None = None):
        self.tol = tol
        self.max_iter_count = max_iter_count
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # shape (p, p), rows are PCs
        self.explained_variance_: np.ndarray | None = None

    def fit(self, X: ArrayLike) -> "IPCA":
        X = _as_2d_float_array(X)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        Q = np.cov(X_centered, rowvar=False)

        if self.components_ is None:
            eigenvalues, eigenvectors = sorted_eigh(Q)
        else:
            eigenvectors = ogita_aishima(
                A=Q,
                X_hat=self.components_.T,
                tol=self.tol,
                max_iter_count=self.max_iter_count,
                sort_by_eigenvalues=True,
            )
            eigenvalues = estimate_eigenvalues(Q, eigenvectors)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors.T
        self.explained_variance_ = eigenvalues
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("IPCA must be fitted before calling transform().")
        X = _as_2d_float_array(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)

    def clear(self) -> None:
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None


class EWMCov:
    """Exponentially weighted moving mean/covariance from Section 5."""

    def __init__(self, alpha: float):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")
        self.alpha = alpha
        self.dim: int | None = None
        self.mean: np.ndarray | None = None  # column vector
        self.cov: np.ndarray | None = None

    def add(self, x: ArrayLike) -> None:
        x = _as_column_vector(x)

        if self.mean is None:
            self.mean = x
            self.dim = x.shape[0]
            self.cov = np.zeros((self.dim, self.dim), dtype=float)
            return

        self.mean = (1.0 - self.alpha) * x + self.alpha * self.mean
        x_centered = x - self.mean
        self.cov = (1.0 - self.alpha) * (x_centered @ x_centered.T) + self.alpha * self.cov


class EWMPCA:
    """
    Exponentially weighted moving PCA from the paper.

    This class supports:
    - online mode: add(x)
    - batch mode: add_all(X)
    """

    def __init__(
        self,
        alpha: float,
        W_initial: ArrayLike | None = None,
        tol: float = 1e-6,
        max_iter_count: int | None = None,
        prime_size: int = 100,
    ):
        self.alpha = alpha
        self._ewmcov = EWMCov(alpha=alpha)
        self.W_: np.ndarray | None = None if W_initial is None else np.asarray(W_initial, dtype=float)
        self.tol = tol
        self.max_iter_count = max_iter_count
        self.prime_size = prime_size

    @property
    def mean_(self) -> np.ndarray | None:
        return None if self._ewmcov.mean is None else self._ewmcov.mean.ravel()

    @property
    def cov_(self) -> np.ndarray | None:
        return self._ewmcov.cov

    def _prime_from_batch(self, X: ArrayLike) -> None:
        X = _as_2d_float_array(X)
        m = min(self.prime_size, X.shape[0])
        if m < 2:
            raise ValueError("Need at least 2 observations to initialize W_initial from data.")
        sample_cov = np.cov(X[:m], rowvar=False)
        _, eigenvectors = sorted_eigh(sample_cov)
        self.W_ = eigenvectors

    def add(self, x: ArrayLike) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError(
                "W_ is not initialized. Provide W_initial or call add_all() first so the model can prime itself."
            )

        x = _as_column_vector(x)
        self._ewmcov.add(x)

        self.W_ = ogita_aishima(
            A=self._ewmcov.cov,
            X_hat=self.W_,
            tol=self.tol,
            max_iter_count=self.max_iter_count,
            sort_by_eigenvalues=True,
        )

        x_centered = x - self._ewmcov.mean
        z = x_centered.reshape(1, -1) @ self.W_
        return z.ravel()

    def add_all(self, X: ArrayLike, verbose: bool = False) -> np.ndarray:
        X = _as_2d_float_array(X)
        if self.W_ is None:
            self._prime_from_batch(X)

        zs = []
        for i, x in enumerate(X):
            if verbose and (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} observations")
            zs.append(self.add(x))
        return np.vstack(zs)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 5))

    ipca = IPCA()
    ipca.fit(X[:100])
    Z_ipca = ipca.transform(X[:100])
    ipca.fit(X[100:200])

    ewmpca = EWMPCA(alpha=0.93)
    Z_ew = ewmpca.add_all(X)

    print("IPCA transformed shape:", Z_ipca.shape)
    print("EWMPCA transformed shape:", Z_ew.shape)
    print("Last EW mean shape:", ewmpca.mean_.shape)

import numpy as np
from ewmpca_from_paper import IPCA, EWMPCA

# X: shape (n_samples, n_features)

# -------------------
# 1) Iterated PCA
# -------------------
ipca = IPCA(tol=1e-6, max_iter_count=None)

# Example: fit sequential windows
X1 = X[:250]
X2 = X[250:500]

ipca.fit(X1)
Z1 = ipca.transform(X1)

ipca.fit(X2)   # reuses previous eigenvectors as warm start
Z2 = ipca.transform(X2)

print(ipca.components_.shape)          # (p, p)
print(ipca.explained_variance_.shape)  # (p,)

# -------------------
# 2) Exponentially Weighted Moving PCA
# -------------------
ewmpca = EWMPCA(alpha=0.93, tol=1e-6, max_iter_count=None, prime_size=100)

Z = ewmpca.add_all(X)   # batch mode
print(Z.shape)          # (n_samples, n_features)

# online mode:
ewmpca_online = EWMPCA(alpha=0.93, W_initial=np.eye(X.shape[1]))
z_t = ewmpca_online.add(X[0])

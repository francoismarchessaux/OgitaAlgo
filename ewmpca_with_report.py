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



def _ensure_1d_float_array(x: ArrayLike, expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"Expected vector of length {expected_len}, got {arr.shape[0]}.")
    return arr


def alpha_from_half_life(half_life: float) -> float:
    if half_life <= 0:
        raise ValueError("half_life must be strictly positive.")
    return float(np.exp(np.log(0.5) / half_life))


def compute_ewmpca_report(
    X: ArrayLike,
    portfolio_exposure: ArrayLike,
    ks: tuple[int, ...] | list[int] = (1, 2, 3, 4, 5),
    alpha: float = 0.97,
    feature_names: list[str] | None = None,
    dates: ArrayLike | None = None,
    tol: float = 1e-6,
    max_iter_count: int | None = None,
    prime_size: int = 100,
    model: EWMPCA | None = None,
) -> dict:
    """
    Build a compact EWMPCA report for a node exposure / portfolio vector.

    Parameters
    ----------
    X : array-like, shape (T, p)
        Time series of node moves, e.g. daily changes in vol at each surface node.
    portfolio_exposure : array-like, shape (p,)
        Exposure vector of the portfolio in the same node space as X.
        Actual PnL is computed as X @ portfolio_exposure.
    ks : iterable of int
        Numbers of PCs to keep in the report.
    alpha : float
        EW decay parameter, ignored if `model` is provided.
    feature_names : list[str], optional
        Names of the p surface nodes.
    dates : array-like, optional
        Dates corresponding to the T rows of X.
    tol, max_iter_count, prime_size
        EWMPCA parameters, ignored if `model` is provided.
    model : EWMPCA, optional
        Existing fitted EWMPCA model. If omitted, a model is fitted on X.

    Returns
    -------
    dict
        Dictionary containing the fitted model, final loadings, explained variance,
        and per-k PnL / hedge-ratio summaries.
    """
    X = _as_2d_float_array(X)
    T, p = X.shape
    portfolio_exposure = _ensure_1d_float_array(portfolio_exposure, expected_len=p)

    unique_ks = sorted({int(k) for k in ks})
    if not unique_ks:
        raise ValueError("ks must contain at least one positive integer.")
    if min(unique_ks) <= 0:
        raise ValueError("All ks must be strictly positive.")
    if max(unique_ks) > p:
        raise ValueError(f"Largest requested k is {max(unique_ks)}, but p={p}.")

    if feature_names is None:
        feature_names = [f"node_{j}" for j in range(p)]
    elif len(feature_names) != p:
        raise ValueError("feature_names must have length equal to the number of columns in X.")

    if dates is None:
        dates = np.arange(T)
    else:
        dates = np.asarray(dates)
        if dates.shape[0] != T:
            raise ValueError("dates must have length equal to the number of rows in X.")

    if model is None:
        model = EWMPCA(
            alpha=alpha,
            tol=tol,
            max_iter_count=max_iter_count,
            prime_size=prime_size,
        )
        scores = model.add_all(X)
    else:
        if model.W_ is None or model.cov_ is None:
            raise ValueError("Provided model must already be fitted.")
        x_centered = X - model.mean_.reshape(1, -1)
        scores = x_centered @ model.W_

    W = np.asarray(model.W_, dtype=float)
    cov = np.asarray(model.cov_, dtype=float)

    eigvals = estimate_eigenvalues(cov, W)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    W = W[:, order]
    scores = scores[:, order]

    eigvals_clipped = np.maximum(eigvals, 0.0)
    total_var = float(np.sum(eigvals_clipped))
    explained_ratio = eigvals_clipped / total_var if total_var > 0 else np.zeros_like(eigvals_clipped)
    cumulative_ratio = np.cumsum(explained_ratio)

    actual_pnl = X @ portfolio_exposure

    per_k = {}
    for k in unique_ks:
        Wk = W[:, :k]

        # coordinates of the portfolio in the PCA basis
        factor_exposure = Wk.T @ portfolio_exposure

        # reconstructed exposure using the first k PCs
        reconstructed_exposure = Wk @ factor_exposure
        residual_exposure = portfolio_exposure - reconstructed_exposure

        pca_pnl = X @ reconstructed_exposure
        error_pnl = actual_pnl - pca_pnl
        hedge_pnl = -pca_pnl
        hedged_realized_pnl = actual_pnl + hedge_pnl

        rms_error = float(np.sqrt(np.mean(error_pnl ** 2)))
        mae_error = float(np.mean(np.abs(error_pnl)))
        var_actual = float(np.var(actual_pnl))
        var_error = float(np.var(error_pnl))
        r2 = float(1.0 - var_error / var_actual) if var_actual > 0 else np.nan

        per_k[k] = {
            "k": k,
            "factor_exposure": factor_exposure,
            "reconstructed_exposure": reconstructed_exposure,
            "residual_exposure": residual_exposure,
            "hedge_ratios_pc_space": -factor_exposure,
            "actual_pnl": actual_pnl,
            "pca_pnl": pca_pnl,
            "error_pnl": error_pnl,
            "hedge_pnl": hedge_pnl,
            "hedged_realized_pnl": hedged_realized_pnl,
            "rmse": rms_error,
            "mae": mae_error,
            "r2": r2,
            "explained_variance_ratio_sum": float(np.sum(explained_ratio[:k])),
        }

    return {
        "model": model,
        "dates": dates,
        "feature_names": feature_names,
        "scores": scores,
        "loadings": W,
        "eigenvalues": eigvals,
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance_ratio": cumulative_ratio,
        "actual_pnl": actual_pnl,
        "portfolio_exposure": portfolio_exposure,
        "per_k": per_k,
    }


def print_ewmpca_report_summary(report: dict) -> None:
    eig = report["eigenvalues"]
    evr = report["explained_variance_ratio"]
    cum = report["cumulative_explained_variance_ratio"]

    print("Final EW explained variance (top 10 PCs):")
    top = min(10, len(eig))
    for i in range(top):
        print(
            f"PC{i+1:>2d} | eigenvalue={eig[i]: .6e} | "
            f"EVR={100*evr[i]:6.2f}% | Cumulative={100*cum[i]:6.2f}%"
        )

    print("\nPnL summary by number of PCs:")
    for k, item in report["per_k"].items():
        print(
            f"k={k:>2d} | EVR={100*item['explained_variance_ratio_sum']:6.2f}% | "
            f"RMSE={item['rmse']: .6e} | MAE={item['mae']: .6e} | R2={item['r2']: .4f}"
        )


def report_to_dataframes(report: dict):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("report_to_dataframes requires pandas.") from exc

    feature_names = report["feature_names"]
    dates = report["dates"]
    loadings = report["loadings"]
    evr = report["explained_variance_ratio"]
    cum = report["cumulative_explained_variance_ratio"]
    scores = report["scores"]

    pc_names = [f"PC{i+1}" for i in range(loadings.shape[1])]

    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=pc_names)
    explained_variance_df = pd.DataFrame(
        {
            "eigenvalue": report["eigenvalues"],
            "explained_variance_ratio": evr,
            "cumulative_explained_variance_ratio": cum,
        },
        index=pc_names,
    )
    scores_df = pd.DataFrame(scores, index=dates, columns=pc_names)

    pnl_dfs = {}
    hedge_ratio_dfs = {}
    for k, item in report["per_k"].items():
        pnl_dfs[k] = pd.DataFrame(
            {
                "actual_pnl": item["actual_pnl"],
                "pca_pnl": item["pca_pnl"],
                "error_pnl": item["error_pnl"],
                "hedge_pnl": item["hedge_pnl"],
                "hedged_realized_pnl": item["hedged_realized_pnl"],
            },
            index=dates,
        )
        hedge_ratio_dfs[k] = pd.DataFrame(
            {
                "portfolio_exposure": report["portfolio_exposure"],
                "reconstructed_exposure": item["reconstructed_exposure"],
                "residual_exposure": item["residual_exposure"],
            },
            index=feature_names,
        )
        for j in range(k):
            hedge_ratio_dfs[k][f"PC{j+1}_loading"] = report["loadings"][:, j]
        hedge_ratio_dfs[k].attrs["hedge_ratios_pc_space"] = item["hedge_ratios_pc_space"]
        hedge_ratio_dfs[k].attrs["factor_exposure"] = item["factor_exposure"]

    return {
        "loadings": loadings_df,
        "explained_variance": explained_variance_df,
        "scores": scores_df,
        "pnl_by_k": pnl_dfs,
        "exposures_by_k": hedge_ratio_dfs,
    }


def plot_ewmpca_report(report: dict, k: int = 3, top_loadings: int = 20) -> None:
    import matplotlib.pyplot as plt

    if k not in report["per_k"]:
        raise ValueError(f"k={k} not found in report['per_k'].")

    dates = report["dates"]
    evr = report["explained_variance_ratio"]
    item = report["per_k"][k]
    loadings = report["loadings"][:, :k]
    feature_names = np.asarray(report["feature_names"])

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, min(10, len(evr)) + 1), 100.0 * evr[:10], marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance (%)")
    plt.title("Final EW explained variance")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(dates, item["actual_pnl"], label="Real PnL")
    plt.plot(dates, item["pca_pnl"], label=f"PCA PnL ({k} PCs)")
    plt.plot(dates, item["error_pnl"], label="Model error")
    plt.legend()
    plt.title(f"PnL decomposition with {k} PCs")
    plt.tight_layout()
    plt.show()

    for j in range(k):
        order = np.argsort(np.abs(loadings[:, j]))[::-1][:top_loadings]
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(order)), loadings[order, j])
        plt.xticks(np.arange(len(order)), feature_names[order], rotation=90)
        plt.title(f"PC{j+1} top {len(order)} loadings")
        plt.tight_layout()
        plt.show()

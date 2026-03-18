import numpy as np

def ogita_aishima_step_fast(A: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    extra = estimate_eigenvalues(A=A, X_hat=X_hat, return_extra=True)
    lambdas = extra["lambda"]
    R = extra["R"]
    S = extra["S"]

    D = np.diag(lambdas)
    delta = 2.0 * (
        np.linalg.norm(S - D, ord=2)
        + np.linalg.norm(A, ord=2) * np.linalg.norm(R, ord=2)
    )

    lam_i = lambdas[:, None]
    lam_j = lambdas[None, :]
    denom = lam_j - lam_i

    mask = np.abs(denom) > delta

    E = 0.5 * R.copy()
    E[mask] = (S[mask] + lam_j[mask] * R[mask]) / denom[mask]

    return X_hat + X_hat @ E

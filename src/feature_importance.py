import numpy as np

from src.evaluate import rmse


def permutation_feature_importance(
    model,
    X_test,
    y_true,
    from_norm,
    feature_names,
    device="cpu",
    repeats=3,
    seed=42,
):
    """
    Estimate feature importance by shuffling one feature at a time
    and measuring how much RMSE gets worse.

    Larger RMSE increase = more important feature.

    Note:
    X_test includes feature columns plus target-history as the final channel.
    This function only tests the original feature columns, not target history.
    """
    import torch

    rng = np.random.default_rng(seed)

    model.eval()

    with torch.no_grad():
        base_pred_norm = model(torch.tensor(X_test).to(device)).cpu().numpy().flatten()

    base_pred = from_norm(base_pred_norm)
    base_score = rmse(y_true, base_pred)

    results = []

    for feature_idx, feature_name in enumerate(feature_names):
        rmse_scores = []

        for _ in range(repeats):
            X_perm = X_test.copy()

            # Shuffle this feature across samples while keeping sequence structure.
            shuffled_order = rng.permutation(X_perm.shape[0])
            X_perm[:, :, feature_idx] = X_perm[shuffled_order, :, feature_idx]

            with torch.no_grad():
                pred_norm = model(torch.tensor(X_perm).to(device)).cpu().numpy().flatten()

            pred = from_norm(pred_norm)
            score = rmse(y_true, pred)
            rmse_scores.append(score)

        avg_rmse = float(np.mean(rmse_scores))
        rmse_increase = avg_rmse - base_score

        results.append(
            {
                "feature": feature_name,
                "base_rmse": base_score,
                "permuted_rmse": avg_rmse,
                "rmse_increase": rmse_increase,
            }
        )

    results = sorted(results, key=lambda row: row["rmse_increase"], reverse=True)

    return results
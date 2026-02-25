"""
MLOps — Clustering Model Training
Trains, evaluates, and persists the K-Means customer segmentation model.

Steps:
  1. Load and compute RFM features from raw data
  2. Train K-Means for k=2..8 (elbow method)
  3. Select best k via silhouette score
  4. Persist model artifact (joblib) + metadata JSON
  5. Emit evaluation report

Run:
    python -m mlops.train_clustering [--k 4] [--report] [--output-dir mlops/artifacts]
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "mlops" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "mlops" / "reports"


# ─── Feature engineering ─────────────────────────────────────────────────────

def compute_rfm_features(
    customers: list[dict],
    transactions: list[dict],
    recency_days: int = 365,
) -> tuple[list[str], list[list[float]]]:
    """
    Compute RFM feature matrix from raw data.
    Returns (customer_ids, feature_matrix) where each row is [recency, frequency, monetary].
    """
    now = datetime.now()
    cutoff = now - timedelta(days=recency_days)

    # Aggregate per customer
    agg: dict[str, dict] = {}
    for txn in transactions:
        cid = txn.get("customer_id")
        try:
            txn_date = datetime.fromisoformat(txn["date"])
        except (KeyError, ValueError):
            continue
        if txn_date < cutoff:
            continue
        if cid not in agg:
            agg[cid] = {"dates": [], "count": 0, "total": 0.0}
        agg[cid]["dates"].append(txn_date)
        agg[cid]["count"] += 1
        agg[cid]["total"] += float(txn.get("amount", 0))

    ids, rows = [], []
    for cid, data in agg.items():
        recency = (now - max(data["dates"])).days
        rows.append([float(recency), float(data["count"]), float(data["total"])])
        ids.append(cid)

    return ids, rows


# ─── Model training ───────────────────────────────────────────────────────────

def _train_single(X_scaled, k: int, random_state: int = 42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else -1.0
    inertia = float(km.inertia_)
    return km, labels, sil, inertia


def train_kmeans(
    customer_ids: list[str],
    feature_matrix: list[list[float]],
    k_range: range = range(2, 9),
    force_k: Optional[int] = None,
) -> dict:
    """
    Train K-Means over k_range and select best k by silhouette score.
    Returns training result dict with model, scaler, and evaluation metrics.
    """
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise RuntimeError("Install scikit-learn and numpy: pip install scikit-learn numpy")

    if len(feature_matrix) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(feature_matrix)}")

    X = np.array(feature_matrix, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in k_range:
        if k >= len(feature_matrix):
            break
        km, labels, sil, inertia = _train_single(X_scaled, k)
        results.append({"k": k, "model": km, "labels": labels,
                         "silhouette": round(sil, 4), "inertia": round(inertia, 2)})
        logger.info(f"k={k}: silhouette={sil:.4f}, inertia={inertia:.2f}")

    if not results:
        raise ValueError("No valid k values produced results")

    # Elbow detection: largest relative inertia drop
    if force_k is not None and any(r["k"] == force_k for r in results):
        best = next(r for r in results if r["k"] == force_k)
        selection_method = "forced"
    else:
        # Select by best silhouette score
        best = max(results, key=lambda r: r["silhouette"])
        selection_method = "silhouette"

    # Build cluster profiles
    X_arr = np.array(feature_matrix)
    cluster_profiles = []
    for cid in range(best["k"]):
        mask = best["labels"] == cid
        members_idx = [i for i, v in enumerate(mask) if v]
        if not members_idx:
            continue
        member_features = X_arr[mask]
        cluster_profiles.append({
            "cluster_id": cid,
            "size": int(mask.sum()),
            "pct_of_total": round(float(mask.sum()) / len(customer_ids) * 100, 1),
            "avg_recency_days": round(float(member_features[:, 0].mean()), 1),
            "avg_frequency": round(float(member_features[:, 1].mean()), 1),
            "avg_monetary": round(float(member_features[:, 2].mean()), 2),
            "sample_customer_ids": [customer_ids[i] for i in members_idx[:5]],
        })

    return {
        "model": best["model"],
        "scaler": scaler,
        "best_k": best["k"],
        "selection_method": selection_method,
        "feature_names": ["recency_days", "frequency", "monetary"],
        "n_samples": len(customer_ids),
        "customer_ids": customer_ids,
        "labels": [int(l) for l in best["labels"]],
        "metrics": {
            "silhouette_score": best["silhouette"],
            "inertia": best["inertia"],
        },
        "elbow_data": [{"k": r["k"], "inertia": r["inertia"], "silhouette": r["silhouette"]}
                       for r in results],
        "cluster_profiles": cluster_profiles,
    }


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_model(training_result: dict, output_dir: Path = ARTIFACTS_DIR) -> Path:
    """Save model artifact (.joblib) and metadata (.json)."""
    try:
        import joblib
    except ImportError:
        raise RuntimeError("Install joblib: pip install joblib")

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    k = training_result["best_k"]

    model_path = output_dir / f"kmeans_k{k}_{ts}.joblib"
    meta_path = output_dir / f"kmeans_k{k}_{ts}.json"
    latest_path = output_dir / "kmeans_latest.joblib"
    latest_meta_path = output_dir / "kmeans_latest.json"

    # Save model bundle (model + scaler together)
    artifact = {
        "model": training_result["model"],
        "scaler": training_result["scaler"],
        "feature_names": training_result["feature_names"],
    }
    joblib.dump(artifact, model_path)
    joblib.dump(artifact, latest_path)  # always update "latest" symlink-equivalent
    logger.info(f"Model artifact saved: {model_path}")

    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_path": str(model_path),
        "best_k": training_result["best_k"],
        "selection_method": training_result["selection_method"],
        "feature_names": training_result["feature_names"],
        "n_samples": training_result["n_samples"],
        "metrics": training_result["metrics"],
        "elbow_data": training_result["elbow_data"],
        "cluster_profiles": training_result["cluster_profiles"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    latest_meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Metadata saved: {meta_path}")
    return model_path


def load_latest_model(artifacts_dir: Path = ARTIFACTS_DIR) -> dict | None:
    """Load the most recently saved model artifact."""
    try:
        import joblib
    except ImportError:
        logger.error("joblib not installed")
        return None

    latest = artifacts_dir / "kmeans_latest.joblib"
    if not latest.exists():
        logger.warning(f"No model found at {latest}")
        return None

    artifact = joblib.load(latest)
    meta_path = artifacts_dir / "kmeans_latest.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    logger.info(f"Loaded model: k={metadata.get('best_k')}, "
                f"trained={metadata.get('trained_at')}")
    return {**artifact, "metadata": metadata}


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train K-Means clustering model on customer RFM data")
    parser.add_argument("--k", type=int, default=None, help="Force specific k (default: auto via silhouette)")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k for search (default: 2)")
    parser.add_argument("--k-max", type=int, default=8, help="Maximum k for search (default: 8)")
    parser.add_argument("--recency-days", type=int, default=365, help="Transaction lookback window")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--report", action="store_true", help="Save evaluation report")
    args = parser.parse_args()

    # Load data
    customers_path = DATA_DIR / "customers.json"
    transactions_path = DATA_DIR / "transactions.json"
    if not customers_path.exists() or not transactions_path.exists():
        logger.error(f"Data files not found in {DATA_DIR}")
        sys.exit(1)

    customers = json.loads(customers_path.read_text())
    transactions = json.loads(transactions_path.read_text())
    logger.info(f"Loaded {len(customers)} customers, {len(transactions)} transactions")

    # Compute features
    print("\n=== Computing RFM Features ===")
    customer_ids, feature_matrix = compute_rfm_features(customers, transactions, args.recency_days)
    print(f"  Customers with transactions in window: {len(customer_ids)}")
    print(f"  Feature matrix shape: {len(feature_matrix)} × 3 (recency, frequency, monetary)")

    if len(customer_ids) < 2:
        print("  ERROR: Insufficient data for clustering (need ≥ 2 customers with transactions)")
        sys.exit(1)

    # Train
    print(f"\n=== Training K-Means (k={args.k_min}..{args.k_max}) ===")
    result = train_kmeans(
        customer_ids=customer_ids,
        feature_matrix=feature_matrix,
        k_range=range(args.k_min, args.k_max + 1),
        force_k=args.k,
    )

    print(f"\n  Best k = {result['best_k']} (by {result['selection_method']})")
    print(f"  Silhouette score: {result['metrics']['silhouette_score']:.4f}  "
          f"(0.5+ = good, 0.7+ = strong)")
    print(f"  Inertia:          {result['metrics']['inertia']:.2f}")

    # Elbow table
    print("\n  Elbow data:")
    print(f"  {'k':>4}  {'Silhouette':>12}  {'Inertia':>12}")
    for row in result["elbow_data"]:
        marker = " ← best" if row["k"] == result["best_k"] else ""
        print(f"  {row['k']:>4}  {row['silhouette']:>12.4f}  {row['inertia']:>12.2f}{marker}")

    # Cluster profiles
    print("\n=== Cluster Profiles ===")
    for cp in result["cluster_profiles"]:
        print(f"  Cluster {cp['cluster_id']} ({cp['size']} customers, {cp['pct_of_total']}%)")
        print(f"    avg_recency={cp['avg_recency_days']}d, "
              f"avg_freq={cp['avg_frequency']}, "
              f"avg_monetary=${cp['avg_monetary']:.2f}")

    # Save
    print(f"\n=== Saving Artifacts → {args.output_dir} ===")
    model_path = save_model(result, output_dir=args.output_dir)
    print(f"  ✓ Model saved: {model_path.name}")

    if args.report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "trained_at": datetime.now().isoformat(),
            "best_k": result["best_k"],
            "metrics": result["metrics"],
            "elbow_data": result["elbow_data"],
            "cluster_profiles": result["cluster_profiles"],
        }
        report_path = REPORTS_DIR / f"training_report_{ts}.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"  ✓ Report saved: {report_path.name}")


if __name__ == "__main__":
    main()

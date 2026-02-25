"""
MLOps — Batch Inference
Runs offline RFM scoring and clustering for all customers, writing results
to mlops/outputs/. Suitable for scheduled nightly jobs or EKS CronJobs.

Steps:
  1. Load latest trained K-Means model from mlops/artifacts/
  2. Compute RFM features for all customers
  3. Assign cluster labels
  4. Compute churn risk scores
  5. Write output: customer_scores_{timestamp}.json
  6. Optionally update the live data store

Run:
    python -m mlops.batch_inference [--update-store] [--output-dir mlops/outputs]
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "mlops" / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "mlops" / "outputs"


# ─── Feature computation ─────────────────────────────────────────────────────

def _compute_rfm(customers: list[dict], transactions: list[dict]) -> dict[str, dict]:
    """Return {customer_id: {recency_days, frequency, monetary}} for all customers."""
    now = datetime.now()
    agg: dict[str, dict] = {}

    for txn in transactions:
        cid = txn.get("customer_id")
        try:
            txn_date = datetime.fromisoformat(txn["date"])
        except (KeyError, ValueError):
            continue
        if cid not in agg:
            agg[cid] = {"dates": [], "count": 0, "total": 0.0}
        agg[cid]["dates"].append(txn_date)
        agg[cid]["count"] += 1
        agg[cid]["total"] += float(txn.get("amount", 0))

    result = {}
    for cid, data in agg.items():
        result[cid] = {
            "recency_days": (now - max(data["dates"])).days,
            "frequency": data["count"],
            "monetary": round(data["total"], 2),
        }

    # Customers with no transactions
    for c in customers:
        cid = c["customer_id"]
        if cid not in result:
            lp = c.get("last_purchase_date")
            if lp:
                try:
                    recency = (now - datetime.fromisoformat(lp)).days
                except ValueError:
                    recency = 9999
            else:
                recency = 9999
            result[cid] = {"recency_days": recency, "frequency": 0, "monetary": 0.0}
    return result


def _classify_segment(r: float, f: float, m: float) -> str:
    """Rule-based segment classification (mirrors segmentation_tools.py)."""
    # Compute quintile-like rough scores (1-5 scale for 4 customers)
    if r <= 30 and f >= 3 and m >= 500:
        return "Champions"
    elif r <= 60 and f >= 2:
        return "Loyal Customers"
    elif r <= 45:
        return "Recent Customers"
    elif f >= 3 and m >= 200:
        return "Potential Loyalists"
    elif r > 90 and f >= 2:
        return "At Risk"
    elif r > 180:
        return "Lost"
    return "Others"


def _churn_risk_score(recency_days: int, threshold_days: int = 90) -> float:
    """Sigmoid churn risk score — matches identify_churn_risk in segmentation_tools."""
    return round(1 / (1 + math.exp(-0.03 * (recency_days - threshold_days))), 4)


def _clv_projection(customer: dict, transactions: list[dict], years: int = 1) -> float:
    """Simple DCF CLV projection matching calculate_clv in crm_loyalty_tools."""
    TIER_RETENTION = {"bronze": 0.55, "silver": 0.70, "gold": 0.82, "platinum": 0.92}
    tier = customer.get("tier", "bronze")
    retention = TIER_RETENTION.get(tier, 0.65)
    discount_rate = 0.10

    cid = customer["customer_id"]
    cust_txns = [t for t in transactions if t["customer_id"] == cid]
    if not cust_txns:
        return 0.0

    signup_str = customer.get("signup_date", "2020-01-01")
    try:
        months_as_customer = max((datetime.now() - datetime.fromisoformat(signup_str)).days / 30, 1)
    except ValueError:
        months_as_customer = 12

    total_spent = sum(float(t.get("amount", 0)) for t in cust_txns)
    avg_monthly_spend = total_spent / months_as_customer
    monthly_discount = (1 + discount_rate) ** (1 / 12) - 1
    monthly_churn = 1 - retention ** (1 / 12)

    clv = 0.0
    for month in range(years * 12):
        survival = (1 - monthly_churn) ** month
        discounted = avg_monthly_spend * survival / (1 + monthly_discount) ** month
        clv += discounted
    return round(clv, 2)


# ─── Batch scoring pipeline ───────────────────────────────────────────────────

class BatchInferencePipeline:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        artifacts_dir: Path = ARTIFACTS_DIR,
        output_dir: Path = OUTPUTS_DIR,
    ):
        self._data_dir = data_dir
        self._artifacts_dir = artifacts_dir
        self._output_dir = output_dir
        self._customers: list[dict] = []
        self._transactions: list[dict] = []
        self._model_artifact: dict | None = None

    def load_data(self) -> "BatchInferencePipeline":
        self._customers = json.loads((self._data_dir / "customers.json").read_text())
        self._transactions = json.loads((self._data_dir / "transactions.json").read_text())
        logger.info(f"Loaded {len(self._customers)} customers, {len(self._transactions)} txns")
        return self

    def load_model(self) -> "BatchInferencePipeline":
        """Load the latest trained clustering model artifact."""
        try:
            import joblib
        except ImportError:
            logger.warning("joblib not installed — skipping cluster assignment")
            return self

        latest = self._artifacts_dir / "kmeans_latest.joblib"
        if latest.exists():
            self._model_artifact = joblib.load(latest)
            meta = self._artifacts_dir / "kmeans_latest.json"
            if meta.exists():
                metadata = json.loads(meta.read_text())
                logger.info(f"Loaded model: k={metadata.get('best_k')}, "
                            f"trained={metadata.get('trained_at')}")
        else:
            logger.warning("No trained model found — cluster labels will be None")
        return self

    def run(self) -> list[dict]:
        """Score all customers and return list of score records."""
        rfm_map = _compute_rfm(self._customers, self._transactions)
        customer_map = {c["customer_id"]: c for c in self._customers}

        # Prepare cluster assignment if model is available
        cluster_labels: dict[str, int] = {}
        if self._model_artifact:
            try:
                import numpy as np
                cids_with_rfm = list(rfm_map.keys())
                X = np.array([[rfm_map[cid]["recency_days"],
                               rfm_map[cid]["frequency"],
                               rfm_map[cid]["monetary"]] for cid in cids_with_rfm])
                scaler = self._model_artifact["scaler"]
                model = self._model_artifact["model"]
                X_scaled = scaler.transform(X)
                labels = model.predict(X_scaled)
                cluster_labels = {cid: int(label) for cid, label in zip(cids_with_rfm, labels)}
            except Exception as e:
                logger.warning(f"Cluster assignment failed: {e}")

        now = datetime.now().isoformat()
        records = []
        for customer in self._customers:
            cid = customer["customer_id"]
            rfm = rfm_map.get(cid, {"recency_days": 9999, "frequency": 0, "monetary": 0.0})
            churn_risk = _churn_risk_score(rfm["recency_days"])
            clv_1yr = _clv_projection(customer, self._transactions, years=1)

            records.append({
                "customer_id": cid,
                "scored_at": now,
                "rfm": rfm,
                "segment": _classify_segment(
                    rfm["recency_days"], rfm["frequency"], rfm["monetary"]
                ),
                "cluster_id": cluster_labels.get(cid),
                "churn_risk_score": churn_risk,
                "churn_risk_level": (
                    "critical" if churn_risk >= 0.8
                    else "high" if churn_risk >= 0.6
                    else "medium" if churn_risk >= 0.4
                    else "low"
                ),
                "clv_1yr_projection": clv_1yr,
                "tier": customer.get("tier"),
                "lifetime_value": customer.get("lifetime_value", 0),
                "points_balance": customer.get("points_balance", 0),
            })

        # Sort by churn risk descending (most at-risk first)
        records.sort(key=lambda x: x["churn_risk_score"], reverse=True)
        return records

    def save(self, records: list[dict]) -> Path:
        """Write scored records to outputs directory."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._output_dir / f"customer_scores_{ts}.json"

        # Summary stats
        avg_churn = round(sum(r["churn_risk_score"] for r in records) / max(len(records), 1), 4)
        segment_counts: dict[str, int] = {}
        for r in records:
            seg = r["segment"]
            segment_counts[seg] = segment_counts.get(seg, 0) + 1

        output = {
            "generated_at": datetime.now().isoformat(),
            "total_customers": len(records),
            "summary": {
                "avg_churn_risk": avg_churn,
                "critical_risk_count": sum(1 for r in records if r["churn_risk_level"] == "critical"),
                "high_risk_count": sum(1 for r in records if r["churn_risk_level"] == "high"),
                "segment_distribution": segment_counts,
            },
            "records": records,
        }
        path.write_text(json.dumps(output, indent=2))
        logger.info(f"Saved {len(records)} scored records to {path}")
        return path

    def update_store(self, records: list[dict]) -> int:
        """Write churn_risk_score and segment back to the live customers.json."""
        customers_path = self._data_dir / "customers.json"
        customers = json.loads(customers_path.read_text())
        score_map = {r["customer_id"]: r for r in records}

        updated = 0
        for c in customers:
            cid = c["customer_id"]
            if cid in score_map:
                r = score_map[cid]
                c["churn_risk_score"] = r["churn_risk_score"]
                c["churn_risk_level"] = r["churn_risk_level"]
                c["rfm_segment"] = r["segment"]
                c["cluster_id"] = r["cluster_id"]
                c["clv_1yr_projection"] = r["clv_1yr_projection"]
                c["last_scored_at"] = r["scored_at"]
                updated += 1

        customers_path.write_text(json.dumps(customers, indent=2))
        logger.info(f"Updated {updated} customer records in customers.json")
        return updated


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch RFM scoring and cluster assignment")
    parser.add_argument("--update-store", action="store_true",
                        help="Write scores back to data/customers.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    pipeline = (
        BatchInferencePipeline(data_dir=args.data_dir, output_dir=args.output_dir)
        .load_data()
        .load_model()
    )

    print("\n=== Running Batch Inference ===")
    records = pipeline.run()
    print(f"  Scored {len(records)} customers")

    # Print summary
    seg_dist: dict[str, int] = {}
    for r in records:
        seg = r["segment"]
        seg_dist[seg] = seg_dist.get(seg, 0) + 1

    print("\n  Segment distribution:")
    for seg, count in sorted(seg_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"    {seg}: {count}")

    critical = [r for r in records if r["churn_risk_level"] == "critical"]
    print(f"\n  Critical churn risk: {len(critical)} customers")
    for r in critical[:5]:
        print(f"    {r['customer_id']}: risk={r['churn_risk_score']:.3f}, "
              f"CLV=${r['lifetime_value']:,.2f}")

    # Save output
    output_path = pipeline.save(records)
    print(f"\n  ✓ Scores saved: {output_path}")

    if args.update_store:
        updated = pipeline.update_store(records)
        print(f"  ✓ Updated {updated} records in customers.json")


if __name__ == "__main__":
    main()

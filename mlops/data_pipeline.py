"""
MLOps — Data Pipeline
Validates, profiles, and monitors data quality for the Customer Intelligence Platform.

Responsibilities:
  - Schema validation against expected field contracts
  - Completeness and type checks
  - Statistical profiling (distributions, outliers)
  - Data drift detection (PSI — Population Stability Index)
  - Summary reports written to mlops/reports/

Run:
    python -m mlops.data_pipeline [--report] [--strict]
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "mlops" / "reports"


# ─── Schema contracts ─────────────────────────────────────────────────────────

CUSTOMER_SCHEMA = {
    "customer_id": str,
    "name": str,
    "tier": str,
    "points_balance": (int, float),
    "lifetime_value": (int, float),
    "last_purchase_date": str,
}

TRANSACTION_SCHEMA = {
    "transaction_id": str,
    "customer_id": str,
    "product_id": str,
    "date": str,
    "amount": (int, float),
    "quantity": (int, float),
    "category": str,
}

PRODUCT_SCHEMA = {
    "product_id": str,
    "name": str,
    "category": str,
    "price": (int, float),
    "avg_rating": (int, float),
}

CAMPAIGN_SCHEMA = {
    "campaign_id": str,
    "name": str,
    "channel": str,
    "status": str,
    "budget": (int, float),
}

VALID_TIERS = {"bronze", "silver", "gold", "platinum"}
VALID_CHANNELS = {"email", "sms", "push", "social", "display", "search"}


# ─── Validators ───────────────────────────────────────────────────────────────

def _validate_schema(records: list[dict], schema: dict[str, Any], entity: str) -> list[str]:
    errors = []
    for i, rec in enumerate(records):
        for field, expected_type in schema.items():
            if field not in rec:
                errors.append(f"{entity}[{i}]: missing required field '{field}'")
            elif not isinstance(rec[field], expected_type):
                errors.append(
                    f"{entity}[{i}] field '{field}': expected {expected_type}, got {type(rec[field]).__name__}"
                )
    return errors


def _validate_dates(records: list[dict], date_field: str, entity: str) -> list[str]:
    errors = []
    for i, rec in enumerate(records):
        val = rec.get(date_field)
        if not val:
            continue
        try:
            datetime.fromisoformat(val)
        except ValueError:
            errors.append(f"{entity}[{i}]: invalid ISO date in '{date_field}': '{val}'")
    return errors


def _validate_referential_integrity(
    transactions: list[dict],
    customers: list[dict],
    products: list[dict],
) -> list[str]:
    errors = []
    customer_ids = {c["customer_id"] for c in customers}
    product_ids = {p["product_id"] for p in products}
    for i, txn in enumerate(transactions):
        cid = txn.get("customer_id")
        pid = txn.get("product_id")
        if cid and cid not in customer_ids:
            errors.append(f"transaction[{i}]: unknown customer_id '{cid}'")
        if pid and pid not in product_ids:
            errors.append(f"transaction[{i}]: unknown product_id '{pid}'")
    return errors


def _validate_business_rules(
    customers: list[dict],
    transactions: list[dict],
) -> list[str]:
    errors = []
    for i, c in enumerate(customers):
        tier = c.get("tier", "")
        if tier not in VALID_TIERS:
            errors.append(f"customer[{i}] '{c.get('customer_id')}': invalid tier '{tier}'")
        if c.get("points_balance", 0) < 0:
            errors.append(f"customer[{i}] '{c.get('customer_id')}': negative points_balance")
        if c.get("lifetime_value", 0) < 0:
            errors.append(f"customer[{i}] '{c.get('customer_id')}': negative lifetime_value")

    for i, txn in enumerate(transactions):
        if txn.get("amount", 0) <= 0:
            errors.append(f"transaction[{i}] '{txn.get('transaction_id')}': non-positive amount")
        if txn.get("quantity", 0) <= 0:
            errors.append(f"transaction[{i}] '{txn.get('transaction_id')}': non-positive quantity")
    return errors


# ─── Statistical profiling ────────────────────────────────────────────────────

def _profile_numeric(values: list[float], name: str) -> dict:
    if not values:
        return {"field": name, "count": 0}
    n = len(values)
    sorted_vals = sorted(values)
    total = sum(values)
    mean = total / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return {
        "field": name,
        "count": n,
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "p25": round(sorted_vals[int(n * 0.25)], 4),
        "p50": round(sorted_vals[int(n * 0.50)], 4),
        "p75": round(sorted_vals[int(n * 0.75)], 4),
        "null_count": 0,
    }


def _compute_psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Population Stability Index — detects distribution shift between two populations.
    PSI < 0.1: stable, 0.1-0.2: slight shift, >0.2: significant drift.
    """
    if not expected or not actual:
        return 0.0

    min_val = min(min(expected), min(actual))
    max_val = max(max(expected), max(actual))
    if min_val == max_val:
        return 0.0

    bin_edges = [min_val + (max_val - min_val) * i / bins for i in range(bins + 1)]

    def bin_counts(data: list[float]) -> list[float]:
        counts = [0] * bins
        for v in data:
            for j in range(bins):
                if bin_edges[j] <= v < bin_edges[j + 1] or (j == bins - 1 and v == bin_edges[-1]):
                    counts[j] += 1
                    break
        total = max(len(data), 1)
        return [max(c / total, 1e-6) for c in counts]  # avoid log(0)

    exp_pct = bin_counts(expected)
    act_pct = bin_counts(actual)
    psi = sum((a - e) * math.log(a / e) for e, a in zip(exp_pct, act_pct))
    return round(psi, 4)


# ─── Main pipeline ────────────────────────────────────────────────────────────

class DataPipeline:
    """Validates, profiles, and drift-checks all data files."""

    def __init__(self, data_dir: Path = DATA_DIR, baseline_dir: Path | None = None):
        self._data_dir = data_dir
        self._baseline_dir = baseline_dir  # for drift comparison
        self._customers: list[dict] = []
        self._transactions: list[dict] = []
        self._products: list[dict] = []
        self._campaigns: list[dict] = []
        self._loyalty_events: list[dict] = []

    def load(self) -> "DataPipeline":
        files = {
            "_customers": "customers.json",
            "_transactions": "transactions.json",
            "_products": "products.json",
            "_campaigns": "campaigns.json",
            "_loyalty_events": "loyalty_events.json",
        }
        for attr, filename in files.items():
            path = self._data_dir / filename
            if path.exists():
                data = json.loads(path.read_text())
                setattr(self, attr, data)
                logger.info(f"Loaded {len(data)} records from {filename}")
            else:
                logger.warning(f"File not found: {path}")
        return self

    def validate(self) -> dict:
        """Run all validation checks. Returns a report dict."""
        errors: list[str] = []
        warnings: list[str] = []

        # Schema validation
        errors += _validate_schema(self._customers, CUSTOMER_SCHEMA, "customer")
        errors += _validate_schema(self._transactions, TRANSACTION_SCHEMA, "transaction")
        errors += _validate_schema(self._products, PRODUCT_SCHEMA, "product")
        errors += _validate_schema(self._campaigns, CAMPAIGN_SCHEMA, "campaign")

        # Date format validation
        errors += _validate_dates(self._customers, "last_purchase_date", "customer")
        errors += _validate_dates(self._transactions, "date", "transaction")

        # Referential integrity
        errors += _validate_referential_integrity(
            self._transactions, self._customers, self._products
        )

        # Business rules
        errors += _validate_business_rules(self._customers, self._transactions)

        # Warnings — data freshness
        now = datetime.now()
        cutoff = now - timedelta(days=365)
        old_txns = [
            t for t in self._transactions
            if datetime.fromisoformat(t["date"]) < cutoff
            if t.get("date")
        ]
        if old_txns:
            warnings.append(
                f"{len(old_txns)} transactions are older than 365 days — "
                "consider archiving stale data"
            )

        # Warnings — low customer count
        if len(self._customers) < 10:
            warnings.append(f"Only {len(self._customers)} customers — model quality may be low")

        return {
            "validated_at": now.isoformat(),
            "record_counts": {
                "customers": len(self._customers),
                "transactions": len(self._transactions),
                "products": len(self._products),
                "campaigns": len(self._campaigns),
                "loyalty_events": len(self._loyalty_events),
            },
            "errors": errors,
            "warnings": warnings,
            "passed": len(errors) == 0,
        }

    def profile(self) -> dict:
        """Compute statistical profiles for key numeric fields."""
        clv_vals = [c.get("lifetime_value", 0) for c in self._customers]
        points_vals = [c.get("points_balance", 0) for c in self._customers]
        txn_amounts = [float(t.get("amount", 0)) for t in self._transactions]
        ratings = [p.get("avg_rating", 0) for p in self._products]
        prices = [p.get("price", 0) for p in self._products]

        # Category distribution
        cat_counts: dict[str, int] = {}
        for t in self._transactions:
            cat = t.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        tier_counts: dict[str, int] = {}
        for c in self._customers:
            t = c.get("tier", "unknown")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        # Purchase recency
        now = datetime.now()
        recency_days = []
        for c in self._customers:
            lp = c.get("last_purchase_date")
            if lp:
                try:
                    recency_days.append((now - datetime.fromisoformat(lp)).days)
                except ValueError:
                    pass

        return {
            "profiled_at": now.isoformat(),
            "numeric_profiles": [
                _profile_numeric(clv_vals, "customer.lifetime_value"),
                _profile_numeric(points_vals, "customer.points_balance"),
                _profile_numeric(txn_amounts, "transaction.amount"),
                _profile_numeric(ratings, "product.avg_rating"),
                _profile_numeric(prices, "product.price"),
                _profile_numeric(recency_days, "customer.recency_days"),
            ],
            "categorical_distributions": {
                "transaction_categories": cat_counts,
                "customer_tiers": tier_counts,
            },
            "avg_transactions_per_customer": round(
                len(self._transactions) / max(len(self._customers), 1), 2
            ),
        }

    def detect_drift(self, baseline_pipeline: "DataPipeline") -> dict:
        """Compare current data distribution vs baseline using PSI.
        PSI < 0.1 = stable; 0.1-0.25 = monitor; >0.25 = alert.
        """
        fields = {
            "lifetime_value": (
                [c.get("lifetime_value", 0) for c in baseline_pipeline._customers],
                [c.get("lifetime_value", 0) for c in self._customers],
            ),
            "points_balance": (
                [c.get("points_balance", 0) for c in baseline_pipeline._customers],
                [c.get("points_balance", 0) for c in self._customers],
            ),
            "transaction_amount": (
                [float(t.get("amount", 0)) for t in baseline_pipeline._transactions],
                [float(t.get("amount", 0)) for t in self._transactions],
            ),
        }

        psi_results = []
        has_drift = False
        for field, (expected, actual) in fields.items():
            psi = _compute_psi(expected, actual)
            status = "stable" if psi < 0.1 else "monitor" if psi < 0.25 else "drift_alert"
            if status == "drift_alert":
                has_drift = True
            psi_results.append({"field": field, "psi": psi, "status": status})

        return {
            "checked_at": datetime.now().isoformat(),
            "psi_scores": psi_results,
            "drift_detected": has_drift,
            "recommendation": (
                "ALERT: Significant data drift detected. Retrain clustering model."
                if has_drift else
                "Data distribution stable. No retraining required."
            ),
        }

    def save_report(self, report: dict, name: str) -> Path:
        """Write a report JSON to mlops/reports/."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"{name}_{ts}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info(f"Report saved: {path}")
        return path


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data pipeline validation and profiling")
    parser.add_argument("--report", action="store_true", help="Save reports to mlops/reports/")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any validation error")
    args = parser.parse_args()

    pipeline = DataPipeline().load()

    print("\n=== Data Validation ===")
    validation = pipeline.validate()
    print(f"  Records:   {validation['record_counts']}")
    if validation["errors"]:
        print(f"  ERRORS ({len(validation['errors'])}):")
        for e in validation["errors"]:
            print(f"    ✗ {e}")
    else:
        print("  ✓ All validation checks passed")

    if validation["warnings"]:
        print(f"  Warnings ({len(validation['warnings'])}):")
        for w in validation["warnings"]:
            print(f"    ⚠ {w}")

    print("\n=== Data Profile ===")
    profile = pipeline.profile()
    for p in profile["numeric_profiles"]:
        print(f"  {p['field']}: mean={p.get('mean', 'N/A')}, std={p.get('std', 'N/A')}, "
              f"p50={p.get('p50', 'N/A')}, count={p.get('count', 0)}")
    print(f"  Tier distribution: {profile['categorical_distributions']['customer_tiers']}")
    print(f"  Avg txns/customer: {profile['avg_transactions_per_customer']}")

    if args.report:
        pipeline.save_report(validation, "validation")
        pipeline.save_report(profile, "profile")

    if args.strict and not validation["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

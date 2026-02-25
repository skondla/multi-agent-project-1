"""
MLOps — Model Registry
Tracks, versions, and manages clustering model artifacts.

Provides:
  - Model registration with metadata (metrics, training params, data stats)
  - Version listing and comparison
  - Promotion (staging → production)
  - Artifact cleanup (keep last N versions)
  - Export model card (Markdown summary)

Run:
    python -m mlops.model_registry list
    python -m mlops.model_registry promote --version <version>
    python -m mlops.model_registry compare --v1 <v1> --v2 <v2>
    python -m mlops.model_registry cleanup --keep 5
    python -m mlops.model_registry card --version <version>
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "mlops" / "artifacts"
REGISTRY_FILE = ARTIFACTS_DIR / "registry.json"


# ─── Registry schema ─────────────────────────────────────────────────────────
#
#  {
#    "models": [
#      {
#        "version": "v1",
#        "trained_at": "...",
#        "model_path": "...",
#        "meta_path": "...",
#        "stage": "staging" | "production" | "archived",
#        "best_k": 4,
#        "n_samples": 20,
#        "metrics": {"silhouette_score": 0.42, "inertia": 123.4},
#        "notes": "...",
#      },
#      ...
#    ]
#  }
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self._dir = artifacts_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._dir / "registry.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())
        return {"models": []}

    def _save(self) -> None:
        self._registry_path.write_text(json.dumps(self._data, indent=2))

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, metadata: dict, notes: str = "") -> str:
        """Register a new model version from a training metadata dict."""
        existing = self._data["models"]
        version = f"v{len(existing) + 1}"

        entry = {
            "version": version,
            "trained_at": metadata.get("trained_at", datetime.now().isoformat()),
            "model_path": metadata.get("model_path", ""),
            "stage": "staging",
            "best_k": metadata.get("best_k"),
            "selection_method": metadata.get("selection_method", "silhouette"),
            "n_samples": metadata.get("n_samples"),
            "feature_names": metadata.get("feature_names", ["recency_days", "frequency", "monetary"]),
            "metrics": metadata.get("metrics", {}),
            "elbow_data": metadata.get("elbow_data", []),
            "cluster_profiles": metadata.get("cluster_profiles", []),
            "notes": notes,
        }
        existing.append(entry)
        self._save()
        logger.info(f"Registered model {version} (k={entry['best_k']}, stage=staging)")
        return version

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_models(self, stage: str | None = None) -> list[dict]:
        models = self._data["models"]
        if stage:
            models = [m for m in models if m["stage"] == stage]
        return models

    def get(self, version: str) -> dict | None:
        for m in self._data["models"]:
            if m["version"] == version:
                return m
        return None

    def get_production(self) -> dict | None:
        for m in reversed(self._data["models"]):
            if m["stage"] == "production":
                return m
        return None

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote(self, version: str, notes: str = "") -> bool:
        """Promote a model version to production. Demotes current production to staging."""
        target = self.get(version)
        if not target:
            logger.error(f"Version {version} not found")
            return False

        # Archive current production
        for m in self._data["models"]:
            if m["stage"] == "production":
                m["stage"] = "archived"
                logger.info(f"Archived previous production model {m['version']}")

        target["stage"] = "production"
        if notes:
            target["notes"] = f"{target.get('notes', '')} | promoted: {notes}".strip(" |")
        target["promoted_at"] = datetime.now().isoformat()

        # Update latest artifact symlink
        model_path = Path(target["model_path"])
        if model_path.exists():
            latest_path = self._dir / "kmeans_latest.joblib"
            shutil.copy2(model_path, latest_path)
            meta_path = model_path.with_suffix(".json")
            if meta_path.exists():
                shutil.copy2(meta_path, self._dir / "kmeans_latest.json")

        self._save()
        logger.info(f"Promoted {version} to production")
        return True

    def archive(self, version: str) -> bool:
        m = self.get(version)
        if not m:
            return False
        m["stage"] = "archived"
        self._save()
        logger.info(f"Archived {version}")
        return True

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(self, v1: str, v2: str) -> dict:
        m1 = self.get(v1)
        m2 = self.get(v2)
        if not m1 or not m2:
            return {"error": f"Version(s) not found: {v1} / {v2}"}

        def metric(m: dict, key: str):
            return m.get("metrics", {}).get(key)

        sil1 = metric(m1, "silhouette_score") or 0
        sil2 = metric(m2, "silhouette_score") or 0
        better = v1 if sil1 > sil2 else v2

        return {
            "v1": {
                "version": v1,
                "k": m1.get("best_k"),
                "silhouette": sil1,
                "inertia": metric(m1, "inertia"),
                "n_samples": m1.get("n_samples"),
                "trained_at": m1.get("trained_at"),
                "stage": m1.get("stage"),
            },
            "v2": {
                "version": v2,
                "k": m2.get("best_k"),
                "silhouette": sil2,
                "inertia": metric(m2, "inertia"),
                "n_samples": m2.get("n_samples"),
                "trained_at": m2.get("trained_at"),
                "stage": m2.get("stage"),
            },
            "winner_by_silhouette": better,
            "silhouette_delta": round(sil1 - sil2, 4),
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self, keep: int = 5) -> int:
        """Archive old models, keeping the most recent `keep` non-archived versions."""
        non_archived = [m for m in self._data["models"] if m["stage"] != "archived"]
        if len(non_archived) <= keep:
            return 0

        to_archive = non_archived[:-keep]  # keep last `keep` by insertion order
        count = 0
        for m in to_archive:
            if m["stage"] != "production":  # never auto-archive production
                m["stage"] = "archived"
                count += 1
        self._save()
        logger.info(f"Archived {count} old model versions")
        return count

    # ── Model card ────────────────────────────────────────────────────────────

    def model_card(self, version: str) -> str:
        m = self.get(version)
        if not m:
            return f"# Model Card: {version} not found"

        sil = m.get("metrics", {}).get("silhouette_score", "N/A")
        inertia = m.get("metrics", {}).get("inertia", "N/A")
        profiles = m.get("cluster_profiles", [])

        cluster_rows = ""
        for cp in profiles:
            cluster_rows += (
                f"| {cp['cluster_id']} | {cp['size']} ({cp['pct_of_total']}%) | "
                f"{cp['avg_recency_days']}d | {cp['avg_frequency']} | "
                f"${cp['avg_monetary']:.2f} |\n"
            )

        return f"""# Model Card — K-Means Customer Segmentation {version}

## Overview
| Field | Value |
|-------|-------|
| **Version** | {version} |
| **Stage** | {m.get('stage', 'unknown')} |
| **Trained at** | {m.get('trained_at', 'N/A')} |
| **Best k** | {m.get('best_k', 'N/A')} |
| **Selection method** | {m.get('selection_method', 'N/A')} |
| **Training samples** | {m.get('n_samples', 'N/A')} |
| **Features** | {', '.join(m.get('feature_names', []))} |

## Evaluation Metrics
| Metric | Value |
|--------|-------|
| **Silhouette Score** | {sil} |
| **Inertia** | {inertia} |

*Silhouette interpretation: >0.7 strong, 0.5–0.7 reasonable, <0.5 weak*

## Cluster Profiles
| Cluster | Size | Avg Recency | Avg Frequency | Avg Monetary |
|---------|------|-------------|---------------|--------------|
{cluster_rows}
## Notes
{m.get('notes', 'None')}

## Intended Use
Customer segmentation for the Customer Intelligence Platform. Cluster labels are used
to route customers to targeted campaigns, personalize recommendations, and prioritize
CRM actions. Not suitable for making credit, hiring, or other consequential decisions.

## Limitations
- Trained on synthetic sample data (20 customers). Retrain with production data.
- K-Means assumes spherical clusters and Euclidean distance on scaled RFM features.
- Clusters may shift seasonally; schedule monthly retraining.
"""


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model registry management")
    sub = parser.add_subparsers(dest="command")

    # list
    list_p = sub.add_parser("list", help="List all registered models")
    list_p.add_argument("--stage", choices=["staging", "production", "archived"])

    # promote
    promo_p = sub.add_parser("promote", help="Promote a model version to production")
    promo_p.add_argument("--version", required=True)
    promo_p.add_argument("--notes", default="")

    # compare
    cmp_p = sub.add_parser("compare", help="Compare two model versions")
    cmp_p.add_argument("--v1", required=True)
    cmp_p.add_argument("--v2", required=True)

    # cleanup
    clean_p = sub.add_parser("cleanup", help="Archive old model versions")
    clean_p.add_argument("--keep", type=int, default=5)

    # card
    card_p = sub.add_parser("card", help="Print model card")
    card_p.add_argument("--version", required=True)

    # auto-register latest artifact
    sub.add_parser("register-latest", help="Register the latest trained artifact")

    args = parser.parse_args()
    registry = ModelRegistry()

    if args.command == "list" or not args.command:
        stage = getattr(args, "stage", None)
        models = registry.list_models(stage=stage)
        if not models:
            print("No models registered.")
            return
        print(f"\n{'Version':>8}  {'Stage':>12}  {'k':>4}  {'Silhouette':>12}  Trained At")
        print("-" * 70)
        for m in models:
            sil = m.get("metrics", {}).get("silhouette_score", "N/A")
            print(f"  {m['version']:>6}  {m['stage']:>12}  {m.get('best_k', '?'):>4}  "
                  f"{str(sil):>12}  {m.get('trained_at', 'N/A')[:19]}")

    elif args.command == "promote":
        ok = registry.promote(args.version, notes=args.notes)
        if ok:
            print(f"✓ Promoted {args.version} to production")
        else:
            print(f"✗ Failed to promote {args.version}")
            sys.exit(1)

    elif args.command == "compare":
        result = registry.compare(args.v1, args.v2)
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        print(f"\n{'Field':>20}  {args.v1:>12}  {args.v2:>12}")
        print("-" * 50)
        for field in ("k", "silhouette", "inertia", "n_samples", "stage"):
            v1_val = result["v1"].get(field, "N/A")
            v2_val = result["v2"].get(field, "N/A")
            print(f"  {field:>18}  {str(v1_val):>12}  {str(v2_val):>12}")
        print(f"\n  Winner by silhouette: {result['winner_by_silhouette']} "
              f"(delta={result['silhouette_delta']:+.4f})")

    elif args.command == "cleanup":
        n = registry.cleanup(keep=args.keep)
        print(f"Archived {n} old versions (keeping ≤{args.keep})")

    elif args.command == "card":
        print(registry.model_card(args.version))

    elif args.command == "register-latest":
        meta_path = ARTIFACTS_DIR / "kmeans_latest.json"
        if not meta_path.exists():
            print("No kmeans_latest.json found. Run train_clustering first.")
            sys.exit(1)
        metadata = json.loads(meta_path.read_text())
        version = registry.register(metadata, notes="Auto-registered after training")
        print(f"✓ Registered as {version}")


if __name__ == "__main__":
    main()

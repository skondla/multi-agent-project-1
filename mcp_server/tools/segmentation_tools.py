"""
Customer Segmentation MCP Tools.
Provides RFM analysis, clustering, churn risk, and segment profiling.

CRITICAL: Never print() to stdout. All logging uses logger (stderr).
"""
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def register(mcp, data_store):
    """Register all segmentation tools onto the FastMCP server instance."""

    @mcp.tool
    def compute_rfm_scores(
        customer_ids: Optional[list] = None,
        recency_days: int = 365,
    ) -> str:
        """
        Compute RFM (Recency, Frequency, Monetary) scores for customers.
        Returns JSON with each customer's R, F, M quintile scores (1-5 scale, 5=best),
        composite RFM score, and segment classification.
        If customer_ids is None, computes for all customers.
        recency_days controls the lookback window for transactions.
        """
        try:
            transactions = data_store.get_transactions()
            customers = data_store.get_customers()
            customer_map = {c["customer_id"]: c for c in customers}

            now = datetime.now()
            cutoff = now - timedelta(days=recency_days)

            if customer_ids:
                target_ids = set(customer_ids)
                transactions = [t for t in transactions if t["customer_id"] in target_ids]

            # Build per-customer RFM raw values
            rfm_data: dict[str, dict] = {}
            for txn in transactions:
                cid = txn["customer_id"]
                try:
                    txn_date = datetime.fromisoformat(txn["date"])
                except (ValueError, KeyError):
                    continue
                if txn_date < cutoff:
                    continue
                if cid not in rfm_data:
                    rfm_data[cid] = {"dates": [], "count": 0, "total": 0.0}
                rfm_data[cid]["dates"].append(txn_date)
                rfm_data[cid]["count"] += 1
                rfm_data[cid]["total"] += float(txn.get("amount", 0))

            if not rfm_data:
                return json.dumps({
                    "status": "success",
                    "total_customers_analyzed": 0,
                    "rfm_data": [],
                    "summary": {"avg_rfm_score": 0, "segments": {}},
                })

            results = []
            for cid, data in rfm_data.items():
                recency = (now - max(data["dates"])).days
                results.append({
                    "customer_id": cid,
                    "name": customer_map.get(cid, {}).get("name", "Unknown"),
                    "tier": customer_map.get(cid, {}).get("tier", "unknown"),
                    "recency_days": recency,
                    "frequency": data["count"],
                    "monetary": round(data["total"], 2),
                })

            # Compute quintile boundaries
            r_vals = sorted([r["recency_days"] for r in results])
            f_vals = sorted([r["frequency"] for r in results])
            m_vals = sorted([r["monetary"] for r in results])

            def percentile(vals, pct):
                if not vals:
                    return 0
                idx = int(len(vals) * pct / 100)
                return vals[min(idx, len(vals) - 1)]

            r_q = [percentile(r_vals, p) for p in [20, 40, 60, 80]]
            f_q = [percentile(f_vals, p) for p in [20, 40, 60, 80]]
            m_q = [percentile(m_vals, p) for p in [20, 40, 60, 80]]

            def score_recency(v):
                # Lower recency (more recent) = better score
                if v <= r_q[0]: return 5
                if v <= r_q[1]: return 4
                if v <= r_q[2]: return 3
                if v <= r_q[3]: return 2
                return 1

            def score_fm(v, q):
                # Higher value = better score
                if v >= q[3]: return 5
                if v >= q[2]: return 4
                if v >= q[1]: return 3
                if v >= q[0]: return 2
                return 1

            for r in results:
                r["r_score"] = score_recency(r["recency_days"])
                r["f_score"] = score_fm(r["frequency"], f_q)
                r["m_score"] = score_fm(r["monetary"], m_q)
                r["rfm_score"] = r["r_score"] + r["f_score"] + r["m_score"]
                r["segment"] = _classify_segment(r["r_score"], r["f_score"], r["m_score"])

            results.sort(key=lambda x: x["rfm_score"], reverse=True)

            segment_counts: dict[str, int] = {}
            for r in results:
                seg = r["segment"]
                segment_counts[seg] = segment_counts.get(seg, 0) + 1

            avg_score = round(sum(r["rfm_score"] for r in results) / len(results), 2)

            return json.dumps({
                "status": "success",
                "total_customers_analyzed": len(results),
                "rfm_data": results[:50],  # Cap output size for context window
                "summary": {
                    "avg_rfm_score": avg_score,
                    "max_possible_score": 15,
                    "segments": segment_counts,
                    "lookback_days": recency_days,
                },
            }, indent=2)

        except Exception as e:
            logger.error(f"compute_rfm_scores error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def run_customer_clustering(
        n_clusters: int = 4,
        features: Optional[list] = None,
    ) -> str:
        """
        Cluster customers using K-Means on RFM features.
        features: list of feature names to use (default: recency_days, frequency, monetary).
        n_clusters: number of clusters (2-8).
        Returns cluster assignments, cluster profiles, and business interpretation.
        """
        try:
            try:
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
            except ImportError:
                return json.dumps({
                    "status": "error",
                    "message": "scikit-learn not installed. Run: pip install scikit-learn numpy"
                })

            n_clusters = max(2, min(8, n_clusters))
            feature_names = features or ["recency_days", "frequency", "monetary"]

            # Compute RFM first
            rfm_raw = json.loads(compute_rfm_scores(recency_days=365))
            if rfm_raw["status"] != "success" or not rfm_raw["rfm_data"]:
                return json.dumps({"status": "error", "message": "Insufficient data for clustering"})

            rfm_list = rfm_raw["rfm_data"]
            if len(rfm_list) < n_clusters:
                return json.dumps({"status": "error", "message": f"Need at least {n_clusters} customers, found {len(rfm_list)}"})

            # Build feature matrix
            valid_features = ["recency_days", "frequency", "monetary", "r_score", "f_score", "m_score"]
            use_features = [f for f in feature_names if f in valid_features]
            if not use_features:
                use_features = ["recency_days", "frequency", "monetary"]

            X = np.array([[row.get(f, 0) for f in use_features] for row in rfm_list])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Build cluster profiles
            cluster_profiles = []
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                members = [rfm_list[i] for i in range(len(rfm_list)) if mask[i]]
                if not members:
                    continue

                avg_recency = round(sum(m["recency_days"] for m in members) / len(members), 1)
                avg_freq = round(sum(m["frequency"] for m in members) / len(members), 1)
                avg_monetary = round(sum(m["monetary"] for m in members) / len(members), 2)
                avg_rfm = round(sum(m["rfm_score"] for m in members) / len(members), 1)

                cluster_profiles.append({
                    "cluster_id": cluster_id,
                    "size": len(members),
                    "pct_of_total": round(len(members) / len(rfm_list) * 100, 1),
                    "avg_recency_days": avg_recency,
                    "avg_frequency": avg_freq,
                    "avg_monetary": avg_monetary,
                    "avg_rfm_score": avg_rfm,
                    "interpretation": _interpret_cluster(avg_recency, avg_freq, avg_monetary),
                    "top_customers": [m["customer_id"] for m in sorted(members, key=lambda x: x["rfm_score"], reverse=True)[:5]],
                })

            customer_assignments = [
                {
                    "customer_id": rfm_list[i]["customer_id"],
                    "name": rfm_list[i]["name"],
                    "cluster_id": int(labels[i]),
                    "rfm_score": rfm_list[i]["rfm_score"],
                }
                for i in range(len(rfm_list))
            ]

            return json.dumps({
                "status": "success",
                "n_clusters": n_clusters,
                "features_used": use_features,
                "total_customers": len(rfm_list),
                "cluster_profiles": cluster_profiles,
                "customer_assignments": customer_assignments[:30],
            }, indent=2)

        except Exception as e:
            logger.error(f"run_customer_clustering error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def get_segment_profile(segment_name: str) -> str:
        """
        Get detailed profile for a named customer segment.
        segment_name: one of Champions, Loyal Customers, Recent Customers,
                      Potential Loyalists, At Risk, Lost, Others.
        Returns avg CLV, top categories, engagement metrics, and recommended actions.
        """
        try:
            rfm_raw = json.loads(compute_rfm_scores(recency_days=365))
            if rfm_raw["status"] != "success":
                return json.dumps({"status": "error", "message": "Could not compute RFM data"})

            # Filter to target segment
            segment_customers = [
                r for r in rfm_raw["rfm_data"]
                if r.get("segment", "").lower() == segment_name.lower()
            ]

            if not segment_customers:
                # Try partial match
                segment_customers = [
                    r for r in rfm_raw["rfm_data"]
                    if segment_name.lower() in r.get("segment", "").lower()
                ]

            if not segment_customers:
                available = list({r.get("segment", "") for r in rfm_raw["rfm_data"]})
                return json.dumps({
                    "status": "not_found",
                    "message": f"Segment '{segment_name}' not found",
                    "available_segments": available,
                })

            customer_ids = [c["customer_id"] for c in segment_customers]
            customers_data = data_store.get_customers(customer_ids)
            customer_map = {c["customer_id"]: c for c in customers_data}

            # Compute category preferences
            transactions = data_store.get_transactions(customer_ids=customer_ids)
            category_spend: dict[str, float] = {}
            for t in transactions:
                cat = t["category"]
                category_spend[cat] = category_spend.get(cat, 0) + float(t.get("amount", 0))

            top_categories = sorted(category_spend.items(), key=lambda x: x[1], reverse=True)[:5]

            avg_clv = round(
                sum(customer_map.get(c["customer_id"], {}).get("lifetime_value", 0) for c in segment_customers) /
                max(len(segment_customers), 1), 2
            )
            avg_recency = round(sum(c["recency_days"] for c in segment_customers) / len(segment_customers), 1)
            avg_freq = round(sum(c["frequency"] for c in segment_customers) / len(segment_customers), 1)
            avg_monetary = round(sum(c["monetary"] for c in segment_customers) / len(segment_customers), 2)

            tier_counts: dict[str, int] = {}
            for c in segment_customers:
                tier = customer_map.get(c["customer_id"], {}).get("tier", "unknown")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            return json.dumps({
                "status": "success",
                "segment_name": segment_name,
                "customer_count": len(segment_customers),
                "pct_of_total": round(len(segment_customers) / max(len(rfm_raw["rfm_data"]), 1) * 100, 1),
                "metrics": {
                    "avg_clv": avg_clv,
                    "avg_recency_days": avg_recency,
                    "avg_frequency": avg_freq,
                    "avg_monetary_365d": avg_monetary,
                },
                "tier_distribution": tier_counts,
                "top_categories": [{"category": cat, "total_spend": round(spend, 2)} for cat, spend in top_categories],
                "customer_ids": customer_ids[:20],
                "recommended_actions": _get_segment_actions(segment_name),
            }, indent=2)

        except Exception as e:
            logger.error(f"get_segment_profile error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def identify_churn_risk(
        threshold_days: int = 90,
        score_threshold: float = 0.5,
    ) -> str:
        """
        Score customers by churn probability using recency and frequency decay.
        threshold_days: days since last purchase to flag as potentially at-risk.
        score_threshold: minimum risk score (0-1) to include in results.
        Returns at-risk customers ranked by risk score with last purchase info.
        """
        try:
            customers = data_store.get_customers()
            now = datetime.now()
            at_risk = []

            for customer in customers:
                last_purchase_str = customer.get("last_purchase_date")
                if not last_purchase_str:
                    continue
                try:
                    last_purchase = datetime.fromisoformat(last_purchase_str)
                except ValueError:
                    continue

                days_inactive = (now - last_purchase).days

                # Risk score: sigmoid-like function based on inactivity
                # 90 days = 0.5 risk, 180 days ≈ 0.8, 270+ days ≈ 0.95
                import math
                risk_score = 1 / (1 + math.exp(-0.03 * (days_inactive - threshold_days)))
                risk_score = round(risk_score, 3)

                if risk_score >= score_threshold:
                    at_risk.append({
                        "customer_id": customer["customer_id"],
                        "name": customer["name"],
                        "tier": customer.get("tier", "unknown"),
                        "lifetime_value": customer.get("lifetime_value", 0),
                        "last_purchase_date": last_purchase_str,
                        "days_since_purchase": days_inactive,
                        "risk_score": risk_score,
                        "risk_level": (
                            "critical" if risk_score >= 0.8
                            else "high" if risk_score >= 0.6
                            else "medium"
                        ),
                    })

            at_risk.sort(key=lambda x: x["risk_score"], reverse=True)

            total_clv_at_risk = round(sum(c["lifetime_value"] for c in at_risk), 2)
            risk_levels = {"critical": 0, "high": 0, "medium": 0}
            for c in at_risk:
                risk_levels[c["risk_level"]] = risk_levels.get(c["risk_level"], 0) + 1

            return json.dumps({
                "status": "success",
                "at_risk_count": len(at_risk),
                "total_clv_at_risk": total_clv_at_risk,
                "risk_level_breakdown": risk_levels,
                "threshold_days": threshold_days,
                "customers": at_risk[:30],
                "recommended_action": (
                    "Launch immediate reactivation campaign with personalized offers "
                    "for critical/high risk customers. Consider win-back incentives "
                    "scaled to customer lifetime value."
                ),
            }, indent=2)

        except Exception as e:
            logger.error(f"identify_churn_risk error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def compare_segments(
        segment_a: str,
        segment_b: str,
        metrics: Optional[list] = None,
    ) -> str:
        """
        Side-by-side comparison of two customer segments.
        segment_a, segment_b: segment names to compare.
        metrics: list of metrics to compare (default: all available).
        Returns comparison table with insights and recommendations.
        """
        try:
            profile_a = json.loads(get_segment_profile(segment_a))
            profile_b = json.loads(get_segment_profile(segment_b))

            if profile_a["status"] not in ("success",) or profile_b["status"] not in ("success",):
                return json.dumps({
                    "status": "error",
                    "message": f"Could not load segment profiles. A: {profile_a.get('message', 'ok')}, B: {profile_b.get('message', 'ok')}",
                })

            def safe_get(profile, *keys, default=0):
                obj = profile
                for k in keys:
                    if not isinstance(obj, dict):
                        return default
                    obj = obj.get(k, default)
                return obj

            comparison = {
                "segment_a": {
                    "name": profile_a["segment_name"],
                    "count": profile_a["customer_count"],
                    "pct_of_total": profile_a["pct_of_total"],
                    "avg_clv": safe_get(profile_a, "metrics", "avg_clv"),
                    "avg_recency_days": safe_get(profile_a, "metrics", "avg_recency_days"),
                    "avg_frequency": safe_get(profile_a, "metrics", "avg_frequency"),
                    "avg_monetary": safe_get(profile_a, "metrics", "avg_monetary_365d"),
                    "top_category": profile_a.get("top_categories", [{}])[0].get("category", "N/A") if profile_a.get("top_categories") else "N/A",
                },
                "segment_b": {
                    "name": profile_b["segment_name"],
                    "count": profile_b["customer_count"],
                    "pct_of_total": profile_b["pct_of_total"],
                    "avg_clv": safe_get(profile_b, "metrics", "avg_clv"),
                    "avg_recency_days": safe_get(profile_b, "metrics", "avg_recency_days"),
                    "avg_frequency": safe_get(profile_b, "metrics", "avg_frequency"),
                    "avg_monetary": safe_get(profile_b, "metrics", "avg_monetary_365d"),
                    "top_category": profile_b.get("top_categories", [{}])[0].get("category", "N/A") if profile_b.get("top_categories") else "N/A",
                },
            }

            # Generate insights
            clv_a = comparison["segment_a"]["avg_clv"]
            clv_b = comparison["segment_b"]["avg_clv"]
            insights = []
            if clv_a > 0 and clv_b > 0:
                pct_diff = round((clv_a - clv_b) / clv_b * 100, 1)
                winner = segment_a if clv_a > clv_b else segment_b
                insights.append(f"{winner} has {abs(pct_diff)}% higher average CLV")

            rec_a = comparison["segment_a"]["avg_recency_days"]
            rec_b = comparison["segment_b"]["avg_recency_days"]
            if rec_a != rec_b:
                more_recent = segment_a if rec_a < rec_b else segment_b
                insights.append(f"{more_recent} is more recently active ({min(rec_a, rec_b):.0f} days avg vs {max(rec_a, rec_b):.0f})")

            return json.dumps({
                "status": "success",
                "comparison": comparison,
                "insights": insights,
                "segment_a_actions": profile_a.get("recommended_actions", []),
                "segment_b_actions": profile_b.get("recommended_actions", []),
            }, indent=2)

        except Exception as e:
            logger.error(f"compare_segments error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    # ─── Private helpers ────────────────────────────────────────────────────

    def _classify_segment(r: int, f: int, m: int) -> str:
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4:
            return "Recent Customers"
        elif f >= 3 and m >= 3:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r == 1:
            return "Lost"
        return "Others"

    def _interpret_cluster(avg_recency: float, avg_freq: float, avg_monetary: float) -> str:
        if avg_recency <= 30 and avg_freq >= 5 and avg_monetary >= 500:
            return "High-Value Active: Frequent, recent, high-spend customers"
        elif avg_recency <= 60 and avg_freq >= 3:
            return "Engaged Regular: Consistent buyers with moderate spend"
        elif avg_recency <= 90 and avg_monetary >= 200:
            return "Occasional Big Spenders: Infrequent but high-value purchases"
        elif avg_recency > 150:
            return "Dormant: Long-inactive customers needing reactivation"
        return "Price-Sensitive: Lower frequency and spend, needs incentives"

    def _get_segment_actions(segment: str) -> list[str]:
        actions = {
            "Champions": [
                "Reward with exclusive VIP perks and early access",
                "Enroll in referral program",
                "Feature as brand ambassadors",
            ],
            "Loyal Customers": [
                "Offer loyalty tier upgrades",
                "Send personalized product recommendations",
                "Provide exclusive member discounts",
            ],
            "Recent Customers": [
                "Nurture with onboarding email series",
                "Cross-sell complementary products",
                "Encourage second purchase with discount",
            ],
            "Potential Loyalists": [
                "Accelerate with bonus points offers",
                "Send personalized content based on preferences",
                "Invite to loyalty program",
            ],
            "At Risk": [
                "Launch immediate win-back campaign",
                "Offer compelling reactivation discount (25-30%)",
                "Personal outreach from customer success",
            ],
            "Lost": [
                "Last-resort win-back with maximum offer",
                "Survey to understand churn reason",
                "Consider removing from active campaigns",
            ],
        }
        return actions.get(segment, ["Analyze purchasing patterns", "Develop targeted engagement strategy"])

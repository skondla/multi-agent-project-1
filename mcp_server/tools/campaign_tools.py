"""
Campaign Optimization MCP Tools.
Provides campaign performance analysis, budget allocation,
audience targeting, ROI forecasting, and A/B test analysis.

CRITICAL: Never print() to stdout. All logging uses logger (stderr).
"""
import json
import logging
import sys
import math
from typing import Optional

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def register(mcp, data_store):
    """Register all campaign optimization tools onto the FastMCP server instance."""

    @mcp.tool
    def get_campaign_performance(
        campaign_ids: Optional[list] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
        """
        Get performance metrics for campaigns.
        campaign_ids: list of campaign IDs (None = all campaigns).
        date_from, date_to: ISO date strings to filter by campaign start date.
        Returns ROI, CTR, conversion rate, revenue, and efficiency metrics.
        """
        try:
            campaigns = data_store.get_campaigns(ids=campaign_ids)

            if date_from:
                campaigns = [c for c in campaigns if c.get("start_date", "") >= date_from]
            if date_to:
                campaigns = [c for c in campaigns if c.get("start_date", "") <= date_to]

            if not campaigns:
                return json.dumps({
                    "status": "success",
                    "message": "No campaigns found for the given criteria",
                    "campaigns": [],
                })

            enriched = []
            for camp in campaigns:
                impressions = camp.get("impressions", 0)
                clicks = camp.get("clicks", 0)
                conversions = camp.get("conversions", 0)
                revenue = camp.get("revenue_generated", 0)
                spend = camp.get("spend", camp.get("budget", 1))

                ctr = round(clicks / impressions * 100, 2) if impressions > 0 else 0
                cvr = round(conversions / clicks * 100, 2) if clicks > 0 else 0
                cpa = round(spend / conversions, 2) if conversions > 0 else 0
                roas = round(revenue / spend, 2) if spend > 0 else 0

                enriched.append({
                    "campaign_id": camp["campaign_id"],
                    "name": camp["name"],
                    "type": camp.get("type", "unknown"),
                    "channel": camp.get("channel", "unknown"),
                    "status": camp.get("status", "unknown"),
                    "target_segment": camp.get("target_segment", "all"),
                    "start_date": camp.get("start_date"),
                    "end_date": camp.get("end_date"),
                    "budget": camp.get("budget", 0),
                    "spend": spend,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "revenue": revenue,
                    "metrics": {
                        "ctr_pct": ctr,
                        "conversion_rate_pct": cvr,
                        "cost_per_acquisition": cpa,
                        "roi": camp.get("roi", roas),
                        "roas": roas,
                        "budget_utilization_pct": round(spend / camp.get("budget", 1) * 100, 1),
                    },
                })

            # Sort by ROI descending
            enriched.sort(key=lambda x: x["metrics"]["roi"], reverse=True)

            # Aggregate totals
            total_spend = sum(c["spend"] for c in enriched)
            total_revenue = sum(c["revenue"] for c in enriched)
            total_conversions = sum(c["conversions"] for c in enriched)
            overall_roi = round(total_revenue / total_spend, 2) if total_spend > 0 else 0

            # Channel breakdown
            channel_perf: dict[str, dict] = {}
            for c in enriched:
                ch = c["channel"]
                if ch not in channel_perf:
                    channel_perf[ch] = {"spend": 0, "revenue": 0, "conversions": 0, "campaigns": 0}
                channel_perf[ch]["spend"] += c["spend"]
                channel_perf[ch]["revenue"] += c["revenue"]
                channel_perf[ch]["conversions"] += c["conversions"]
                channel_perf[ch]["campaigns"] += 1

            for ch in channel_perf:
                s = channel_perf[ch]["spend"]
                r = channel_perf[ch]["revenue"]
                channel_perf[ch]["avg_roi"] = round(r / s, 2) if s > 0 else 0

            return json.dumps({
                "status": "success",
                "total_campaigns": len(enriched),
                "campaigns": enriched,
                "totals": {
                    "total_spend": round(total_spend, 2),
                    "total_revenue": round(total_revenue, 2),
                    "total_conversions": total_conversions,
                    "overall_roi": overall_roi,
                },
                "channel_breakdown": channel_perf,
                "top_performer": enriched[0]["name"] if enriched else None,
                "bottom_performer": enriched[-1]["name"] if len(enriched) > 1 else None,
            }, indent=2)

        except Exception as e:
            logger.error(f"get_campaign_performance error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def recommend_budget_allocation(
        total_budget: float,
        channels: Optional[list] = None,
        objective: str = "maximize_roi",
    ) -> str:
        """
        Recommend optimal budget allocation across channels based on historical ROI.
        total_budget: total budget to allocate in dollars.
        channels: list of channels to consider (default: all historical channels).
        objective: 'maximize_roi', 'maximize_reach', or 'maximize_conversions'.
        Returns allocation per channel with expected returns.
        """
        try:
            campaigns = data_store.get_campaigns(status="completed")
            if not campaigns:
                campaigns = data_store.get_campaigns()

            # Compute per-channel historical metrics
            channel_metrics: dict[str, dict] = {}
            for camp in campaigns:
                ch = camp.get("channel", "unknown")
                if channels and ch not in channels:
                    continue
                if ch not in channel_metrics:
                    channel_metrics[ch] = {
                        "total_spend": 0, "total_revenue": 0,
                        "total_conversions": 0, "total_impressions": 0,
                        "campaign_count": 0,
                    }
                channel_metrics[ch]["total_spend"] += camp.get("spend", 0)
                channel_metrics[ch]["total_revenue"] += camp.get("revenue_generated", 0)
                channel_metrics[ch]["total_conversions"] += camp.get("conversions", 0)
                channel_metrics[ch]["total_impressions"] += camp.get("impressions", 0)
                channel_metrics[ch]["campaign_count"] += 1

            if not channel_metrics:
                return json.dumps({"status": "error", "message": "No historical campaign data available"})

            # Compute efficiency scores per channel
            channel_scores = []
            for ch, m in channel_metrics.items():
                spend = m["total_spend"]
                if spend == 0:
                    continue
                roi = m["total_revenue"] / spend
                cpa = spend / m["total_conversions"] if m["total_conversions"] > 0 else float("inf")
                cpm = spend / (m["total_impressions"] / 1000) if m["total_impressions"] > 0 else float("inf")

                if objective == "maximize_roi":
                    score = roi
                elif objective == "maximize_reach":
                    score = m["total_impressions"] / spend if spend > 0 else 0
                else:  # maximize_conversions
                    score = m["total_conversions"] / spend if spend > 0 else 0

                channel_scores.append({
                    "channel": ch,
                    "score": score,
                    "historical_roi": round(roi, 2),
                    "historical_cpa": round(cpa, 2) if cpa != float("inf") else None,
                    "campaign_count": m["campaign_count"],
                })

            if not channel_scores:
                return json.dumps({"status": "error", "message": "No scored channels available"})

            # Weighted allocation: softmax-like distribution
            total_score = sum(max(cs["score"], 0.001) for cs in channel_scores)
            allocations = []
            for cs in channel_scores:
                weight = max(cs["score"], 0.001) / total_score
                amount = round(total_budget * weight, 2)
                expected_roi = cs["historical_roi"]
                expected_return = round(amount * expected_roi, 2)

                allocations.append({
                    "channel": cs["channel"],
                    "allocation": amount,
                    "allocation_pct": round(weight * 100, 1),
                    "expected_roi": expected_roi,
                    "expected_return": expected_return,
                    "historical_cpa": cs.get("historical_cpa"),
                    "rationale": f"Based on {cs['campaign_count']} historical campaigns with {expected_roi:.1f}x ROI",
                })

            allocations.sort(key=lambda x: x["allocation"], reverse=True)
            total_expected = round(sum(a["expected_return"] for a in allocations), 2)
            blended_roi = round(total_expected / total_budget, 2) if total_budget > 0 else 0

            return json.dumps({
                "status": "success",
                "total_budget": total_budget,
                "objective": objective,
                "allocations": allocations,
                "summary": {
                    "total_expected_return": total_expected,
                    "blended_expected_roi": blended_roi,
                    "top_channel": allocations[0]["channel"] if allocations else None,
                },
            }, indent=2)

        except Exception as e:
            logger.error(f"recommend_budget_allocation error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def identify_target_audience(
        campaign_type: str,
        segment_filter: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """
        Find the best-fit customers for a given campaign type.
        campaign_type: 'promotional', 'retention', 'upsell', 'reactivation', 'loyalty'.
        segment_filter: optional tier filter ('bronze', 'silver', 'gold', 'platinum').
        limit: max number of customers to return.
        Returns scored customer list with estimated response likelihood.
        """
        try:
            customers = data_store.get_customers()
            if segment_filter:
                customers = [c for c in customers if c.get("tier", "").lower() == segment_filter.lower()]

            from datetime import datetime
            now = datetime.now()

            scored_customers = []
            for customer in customers:
                last_purchase_str = customer.get("last_purchase_date", "2020-01-01")
                try:
                    last_purchase = datetime.fromisoformat(last_purchase_str)
                    days_inactive = (now - last_purchase).days
                except ValueError:
                    days_inactive = 999

                clv = customer.get("lifetime_value", 0)
                points = customer.get("points_balance", 0)
                tier = customer.get("tier", "bronze")

                # Score based on campaign type
                if campaign_type == "promotional":
                    # Active, high-value customers
                    score = (1 / max(days_inactive, 1)) * 1000 + clv / 100
                elif campaign_type == "retention":
                    # Medium-risk customers (30-90 days inactive)
                    risk_score = 1 / (1 + math.exp(-0.03 * (days_inactive - 60)))
                    score = risk_score * clv / 100
                elif campaign_type == "upsell":
                    # Recent purchasers with spending potential
                    recency_boost = max(0, 1 - days_inactive / 90)
                    score = recency_boost * clv / 100
                elif campaign_type == "reactivation":
                    # Dormant but high-CLV customers
                    dormant_score = 1 / (1 + math.exp(-0.02 * (days_inactive - 90)))
                    score = dormant_score * clv / 50
                elif campaign_type == "loyalty":
                    # Customers with points activity or near tier upgrade
                    tier_weight = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4}.get(tier, 1)
                    score = (points / 1000 + tier_weight) * (1 / max(days_inactive, 1)) * 100
                else:
                    score = clv / 100

                scored_customers.append({
                    "customer_id": customer["customer_id"],
                    "name": customer["name"],
                    "tier": tier,
                    "lifetime_value": clv,
                    "days_since_purchase": days_inactive,
                    "preferred_channel": customer.get("preferred_channel", "email"),
                    "estimated_response_score": round(score, 3),
                })

            scored_customers.sort(key=lambda x: x["estimated_response_score"], reverse=True)
            top_customers = scored_customers[:limit]

            tier_breakdown: dict[str, int] = {}
            for c in top_customers:
                t = c["tier"]
                tier_breakdown[t] = tier_breakdown.get(t, 0) + 1

            channel_pref: dict[str, int] = {}
            for c in top_customers:
                ch = c["preferred_channel"]
                channel_pref[ch] = channel_pref.get(ch, 0) + 1

            return json.dumps({
                "status": "success",
                "campaign_type": campaign_type,
                "segment_filter": segment_filter,
                "audience_size": len(top_customers),
                "customers": top_customers,
                "tier_breakdown": tier_breakdown,
                "preferred_channels": channel_pref,
                "recommended_channel": max(channel_pref, key=channel_pref.get) if channel_pref else "email",
                "estimated_response_rate": f"{min(round(len(top_customers) * 0.12, 1), 100)}%",
            }, indent=2)

        except Exception as e:
            logger.error(f"identify_target_audience error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def forecast_campaign_roi(
        channel: str,
        budget: float,
        target_segment: str,
        campaign_type: str = "promotional",
    ) -> str:
        """
        Predict expected ROI for a proposed campaign based on historical performance.
        channel: campaign channel ('email', 'sms', 'push').
        budget: proposed campaign budget in dollars.
        target_segment: target customer tier or segment name.
        campaign_type: 'promotional', 'retention', 'loyalty', 'upsell', 'reactivation'.
        Returns forecasted ROI with confidence interval and comparable campaigns.
        """
        try:
            # Find comparable historical campaigns
            all_campaigns = data_store.get_campaigns()
            comparable = [
                c for c in all_campaigns
                if c.get("channel") == channel
                and c.get("status") == "completed"
                and c.get("type") == campaign_type
            ]

            if not comparable:
                # Fall back to same channel
                comparable = [c for c in all_campaigns if c.get("channel") == channel and c.get("status") == "completed"]

            if not comparable:
                # Use all completed
                comparable = [c for c in all_campaigns if c.get("status") == "completed"]

            if not comparable:
                return json.dumps({
                    "status": "no_data",
                    "message": "Insufficient historical data for forecast",
                    "forecasted_roi": 3.0,
                    "confidence": "low",
                })

            rois = [c.get("roi", 0) for c in comparable if c.get("roi", 0) > 0]
            if not rois:
                return json.dumps({"status": "error", "message": "No valid ROI data found"})

            avg_roi = sum(rois) / len(rois)
            min_roi = min(rois)
            max_roi = max(rois)

            # Adjust for segment
            segment_multipliers = {
                "platinum": 1.3, "gold": 1.1, "silver": 0.95,
                "bronze": 0.8, "at_risk": 0.7, "all": 1.0,
                "champions": 1.4, "loyal": 1.2,
            }
            multiplier = segment_multipliers.get(target_segment.lower(), 1.0)
            adjusted_roi = round(avg_roi * multiplier, 2)

            # Estimate outcomes
            expected_revenue = round(budget * adjusted_roi, 2)
            revenue_low = round(budget * min_roi * multiplier, 2)
            revenue_high = round(budget * max_roi * multiplier, 2)

            confidence = "high" if len(comparable) >= 3 else "medium" if len(comparable) >= 2 else "low"

            return json.dumps({
                "status": "success",
                "proposed_campaign": {
                    "channel": channel,
                    "budget": budget,
                    "target_segment": target_segment,
                    "campaign_type": campaign_type,
                },
                "forecast": {
                    "forecasted_roi": adjusted_roi,
                    "expected_revenue": expected_revenue,
                    "revenue_range": {"low": revenue_low, "high": revenue_high},
                    "expected_net_profit": round(expected_revenue - budget, 2),
                    "confidence": confidence,
                    "comparable_campaigns_used": len(comparable),
                },
                "assumptions": [
                    f"Based on {len(comparable)} historical {channel} campaigns",
                    f"Segment '{target_segment}' multiplier: {multiplier}x",
                    f"Historical ROI range: {min_roi:.1f}x - {max_roi:.1f}x",
                    "Market conditions assumed similar to historical periods",
                ],
                "comparable_campaigns": [
                    {
                        "name": c["name"],
                        "roi": c.get("roi", 0),
                        "channel": c["channel"],
                        "target_segment": c.get("target_segment"),
                    }
                    for c in comparable[:5]
                ],
            }, indent=2)

        except Exception as e:
            logger.error(f"forecast_campaign_roi error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def ab_test_analysis(
        control_metric: float,
        test_metric: float,
        control_size: int,
        test_size: int,
        metric_name: str = "conversion_rate",
    ) -> str:
        """
        Analyze A/B test results for statistical significance.
        control_metric: metric value for control group (e.g., 0.05 for 5% conversion rate).
        test_metric: metric value for test group.
        control_size: number of users in control group.
        test_size: number of users in test group.
        metric_name: name of the metric being tested.
        Returns p-value, lift, significance, and recommendation.
        """
        try:
            import math

            if control_metric <= 0 or test_metric <= 0:
                return json.dumps({"status": "error", "message": "Metrics must be positive values"})

            # Compute lift
            lift_pct = round((test_metric - control_metric) / control_metric * 100, 2)

            # Two-proportion z-test
            p_pooled = (control_metric * control_size + test_metric * test_size) / (control_size + test_size)

            if p_pooled <= 0 or p_pooled >= 1:
                # For non-proportion metrics, use approximation
                se = math.sqrt((control_metric**2 / control_size) + (test_metric**2 / test_size))
                if se == 0:
                    z_score = 0
                else:
                    z_score = abs(test_metric - control_metric) / se
            else:
                se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / control_size + 1 / test_size))
                z_score = abs(test_metric - control_metric) / se if se > 0 else 0

            # Approximate p-value from z-score
            def normal_cdf(z):
                return 0.5 * (1 + math.erf(z / math.sqrt(2)))

            p_value = round(2 * (1 - normal_cdf(z_score)), 4)
            significant = p_value < 0.05

            ci_margin = 1.96 * math.sqrt(
                control_metric * (1 - min(control_metric, 0.99)) / control_size +
                test_metric * (1 - min(test_metric, 0.99)) / test_size
            ) if 0 < control_metric < 1 else 0

            if significant and test_metric > control_metric:
                recommendation = f"IMPLEMENT TEST VARIANT: {lift_pct:+.1f}% lift is statistically significant (p={p_value}). Roll out to full audience."
            elif significant and test_metric < control_metric:
                recommendation = f"KEEP CONTROL: Test variant performs worse by {abs(lift_pct):.1f}% (p={p_value}). Do not implement."
            else:
                recommendation = f"INCONCLUSIVE: {lift_pct:+.1f}% difference is not statistically significant (p={p_value}). Run test longer or increase sample size."

            return json.dumps({
                "status": "success",
                "metric_name": metric_name,
                "results": {
                    "control": {"metric": control_metric, "size": control_size},
                    "test": {"metric": test_metric, "size": test_size},
                    "lift_pct": lift_pct,
                    "z_score": round(z_score, 3),
                    "p_value": p_value,
                    "significant_at_95pct": significant,
                    "confidence_interval_95": f"[{round(lift_pct - ci_margin * 100, 1)}%, {round(lift_pct + ci_margin * 100, 1)}%]",
                },
                "recommendation": recommendation,
                "required_sample_size": _compute_required_sample(control_metric),
                "test_adequate": control_size >= 100 and test_size >= 100,
            }, indent=2)

        except Exception as e:
            logger.error(f"ab_test_analysis error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    def _compute_required_sample(baseline_rate: float) -> int:
        """Approximate sample size per group for 80% power at 5% significance."""
        import math
        if baseline_rate <= 0 or baseline_rate >= 1:
            return 1000
        mde = 0.1  # minimum detectable effect: 10% relative lift
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        p_bar = (p1 + p2) / 2
        n = (1.96 + 0.842) ** 2 * (p_bar * (1 - p_bar) * 2) / (p2 - p1) ** 2
        return int(math.ceil(n))

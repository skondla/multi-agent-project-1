"""
CRM and Loyalty Management MCP Tools.
Provides customer profiles, CLV calculation, loyalty summary,
points transactions, and tier upgrade analysis.

CRITICAL: Never print() to stdout. All logging uses logger (stderr).
"""
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

# Tier multipliers for CLV projection
TIER_RETENTION_RATES = {
    "bronze": 0.55,
    "silver": 0.70,
    "gold": 0.82,
    "platinum": 0.92,
}

LOYALTY_REWARDS_CATALOG = [
    {"reward_id": "RWD001", "name": "$10 Store Credit", "points_required": 500, "category": "discount"},
    {"reward_id": "RWD002", "name": "$25 Store Credit", "points_required": 1200, "category": "discount"},
    {"reward_id": "RWD003", "name": "$50 Gift Card", "points_required": 2000, "category": "gift_card"},
    {"reward_id": "RWD004", "name": "Free Shipping (3 months)", "points_required": 1500, "category": "shipping"},
    {"reward_id": "RWD005", "name": "Free Shipping (1 year)", "points_required": 5000, "category": "shipping"},
    {"reward_id": "RWD006", "name": "15% Off Next Order", "points_required": 800, "category": "discount"},
    {"reward_id": "RWD007", "name": "Early Access Pass", "points_required": 1000, "category": "vip"},
    {"reward_id": "RWD008", "name": "Premium Product (up to $75)", "points_required": 3000, "category": "product"},
    {"reward_id": "RWD009", "name": "VIP Experience Reward", "points_required": 5000, "category": "vip"},
    {"reward_id": "RWD010", "name": "Donate to Charity ($20)", "points_required": 1000, "category": "charity"},
]


def register(mcp, data_store):
    """Register all CRM/loyalty tools onto the FastMCP server instance."""

    @mcp.tool
    def get_customer_profile(customer_id: str) -> str:
        """
        Get a comprehensive customer profile including demographics,
        transaction summary, category preferences, loyalty status,
        and engagement metrics.
        customer_id: the customer ID to look up.
        Returns full profile with purchase history summary and recommendations.
        """
        try:
            customer = data_store.get_customer(customer_id)
            if not customer:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            # Transaction analysis
            transactions = data_store.get_customer_transactions(customer_id)
            total_spent = sum(float(t.get("amount", 0)) for t in transactions)
            total_orders = len(transactions)

            # Category breakdown
            category_spend: dict[str, float] = {}
            category_count: dict[str, int] = {}
            for t in transactions:
                cat = t["category"]
                category_spend[cat] = category_spend.get(cat, 0) + float(t.get("amount", 0))
                category_count[cat] = category_count.get(cat, 0) + 1

            top_categories = sorted(
                [{"category": cat, "spend": round(spend, 2), "orders": category_count[cat]}
                 for cat, spend in category_spend.items()],
                key=lambda x: x["spend"],
                reverse=True,
            )[:5]

            # Recent transactions
            recent_txns = transactions[:5]
            recent_products = []
            for t in recent_txns:
                product = data_store.get_product(t["product_id"])
                recent_products.append({
                    "date": t["date"],
                    "product_name": product.get("name", "Unknown") if product else "Unknown",
                    "category": t["category"],
                    "amount": t["amount"],
                })

            # Loyalty info
            loyalty_events = data_store.get_loyalty_events(customer_id)
            total_earned = sum(e["points"] for e in loyalty_events if e["points"] > 0)
            total_redeemed = abs(sum(e["points"] for e in loyalty_events if e["points"] < 0))

            # Days as customer
            signup_str = customer.get("signup_date", "2020-01-01")
            try:
                signup_date = datetime.fromisoformat(signup_str)
                days_as_customer = (datetime.now() - signup_date).days
            except ValueError:
                days_as_customer = 0

            last_purchase_str = customer.get("last_purchase_date", "2020-01-01")
            try:
                last_purchase = datetime.fromisoformat(last_purchase_str)
                days_since_purchase = (datetime.now() - last_purchase).days
            except ValueError:
                days_since_purchase = 999

            avg_order_value = round(total_spent / total_orders, 2) if total_orders > 0 else 0
            purchase_frequency = round(total_orders / max(days_as_customer / 30, 1), 1)

            # Available rewards
            points = customer.get("points_balance", 0)
            available_rewards = [
                r for r in LOYALTY_REWARDS_CATALOG
                if r["points_required"] <= points
            ]

            return json.dumps({
                "status": "success",
                "profile": {
                    "customer_id": customer_id,
                    "name": customer.get("name"),
                    "email": customer.get("email"),
                    "age": customer.get("age"),
                    "gender": customer.get("gender"),
                    "location": f"{customer.get('city')}, {customer.get('state')}",
                    "signup_date": signup_str,
                    "days_as_customer": days_as_customer,
                    "preferred_channel": customer.get("preferred_channel"),
                },
                "loyalty": {
                    "tier": customer.get("tier"),
                    "points_balance": points,
                    "total_points_earned": total_earned,
                    "total_points_redeemed": total_redeemed,
                    "available_rewards_count": len(available_rewards),
                },
                "transaction_summary": {
                    "total_orders": total_orders,
                    "total_spent": round(total_spent, 2),
                    "avg_order_value": avg_order_value,
                    "purchase_frequency_per_month": purchase_frequency,
                    "days_since_last_purchase": days_since_purchase,
                    "lifetime_value": customer.get("lifetime_value", 0),
                },
                "top_categories": top_categories,
                "recent_purchases": recent_products,
            }, indent=2)

        except Exception as e:
            logger.error(f"get_customer_profile error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def calculate_clv(
        customer_id: str,
        horizon_years: int = 3,
        discount_rate: float = 0.10,
    ) -> str:
        """
        Calculate Customer Lifetime Value using a DCF-based model.
        Projects future purchase value based on historical behavior and tier retention.
        customer_id: target customer.
        horizon_years: projection period in years (1, 3, or 5).
        discount_rate: annual discount rate for NPV calculation (default 10%).
        Returns CLV projections for 1/3/5 years with percentile rank among all customers.
        """
        try:
            horizon_years = max(1, min(5, horizon_years))
            customer = data_store.get_customer(customer_id)
            if not customer:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            # Historical metrics
            transactions = data_store.get_customer_transactions(customer_id)
            total_spent = sum(float(t.get("amount", 0)) for t in transactions)

            signup_str = customer.get("signup_date", "2020-01-01")
            try:
                signup_date = datetime.fromisoformat(signup_str)
                months_as_customer = max((datetime.now() - signup_date).days / 30, 1)
            except ValueError:
                months_as_customer = 12

            avg_monthly_spend = total_spent / months_as_customer
            tier = customer.get("tier", "bronze")
            retention_rate = TIER_RETENTION_RATES.get(tier, 0.65)

            # DCF CLV projection
            def project_clv(years: int) -> float:
                total = 0.0
                monthly_spend = avg_monthly_spend
                monthly_discount = (1 + discount_rate) ** (1 / 12) - 1
                monthly_churn = 1 - retention_rate ** (1 / 12)

                for month in range(years * 12):
                    # Probability customer is still active
                    survival = (1 - monthly_churn) ** month
                    # Discounted value
                    discounted = monthly_spend * survival / (1 + monthly_discount) ** month
                    total += discounted

                return round(total, 2)

            clv_1yr = project_clv(1)
            clv_3yr = project_clv(3)
            clv_5yr = project_clv(5)
            clv_horizon = project_clv(horizon_years)

            # Percentile rank among all customers
            all_customers = data_store.get_customers()
            all_clvs = [c.get("lifetime_value", 0) for c in all_customers]
            all_clvs.sort()
            current_clv = customer.get("lifetime_value", 0)
            rank = sum(1 for v in all_clvs if v <= current_clv)
            percentile = round(rank / max(len(all_clvs), 1) * 100, 1)

            # Segment classification by CLV
            if current_clv >= 10000:
                clv_segment = "VIP (Top 10%)"
            elif current_clv >= 3000:
                clv_segment = "High Value (Top 25%)"
            elif current_clv >= 1000:
                clv_segment = "Mid Value"
            else:
                clv_segment = "Entry Level"

            return json.dumps({
                "status": "success",
                "customer_id": customer_id,
                "customer_name": customer.get("name"),
                "tier": tier,
                "current_lifetime_value": current_clv,
                "clv_percentile": percentile,
                "clv_segment": clv_segment,
                "projections": {
                    "clv_1_year": clv_1yr,
                    "clv_3_year": clv_3yr,
                    "clv_5_year": clv_5yr,
                    f"clv_{horizon_years}_year": clv_horizon,
                },
                "model_inputs": {
                    "avg_monthly_spend": round(avg_monthly_spend, 2),
                    "annual_retention_rate": retention_rate,
                    "discount_rate": discount_rate,
                    "months_of_history": round(months_as_customer, 1),
                },
                "assumptions": [
                    f"Retention rate based on {tier} tier: {retention_rate:.0%} annually",
                    f"Average monthly spend: ${avg_monthly_spend:.2f} (based on {len(transactions)} transactions)",
                    "No external growth or inflation adjustments applied",
                    f"Discount rate: {discount_rate:.0%} per year",
                ],
            }, indent=2)

        except Exception as e:
            logger.error(f"calculate_clv error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def get_loyalty_summary(customer_id: str) -> str:
        """
        Get a comprehensive loyalty program summary for a customer.
        Includes current tier, points balance, recent events, available rewards,
        and progress toward next tier.
        customer_id: target customer.
        """
        try:
            customer = data_store.get_customer(customer_id)
            if not customer:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            tier = customer.get("tier", "bronze")
            points = customer.get("points_balance", 0)

            # Loyalty events
            events = data_store.get_loyalty_events(customer_id)
            events_sorted = sorted(events, key=lambda x: x.get("date", ""), reverse=True)

            total_earned = sum(e["points"] for e in events if e["points"] > 0)
            total_redeemed = abs(sum(e["points"] for e in events if e["points"] < 0))

            # Points by type
            bonus_points = sum(e["points"] for e in events if e.get("event_type") == "bonus_points")
            purchase_points = sum(e["points"] for e in events if e.get("event_type") == "points_earned")

            # Tier progress
            thresholds = data_store.get_tier_thresholds()
            next_tier = data_store.get_next_tier(tier)
            current_min = thresholds.get(tier, {}).get("min", 0)

            if next_tier:
                next_min = thresholds[next_tier]["min"]
                points_to_next = max(0, next_min - points)
                progress_pct = round(
                    min((points - current_min) / max(next_min - current_min, 1) * 100, 100),
                    1,
                )
            else:
                points_to_next = 0
                progress_pct = 100.0

            # Available rewards
            available = [r for r in LOYALTY_REWARDS_CATALOG if r["points_required"] <= points]
            unavailable_close = [
                r for r in LOYALTY_REWARDS_CATALOG
                if points < r["points_required"] <= points + 1000
            ]

            # Tier benefits
            tier_benefits = {
                "bronze": ["Earn 1 point per $1 spent", "Birthday bonus points"],
                "silver": ["Earn 1.5 points per $1 spent", "Birthday bonus points", "Free standard shipping on orders $50+"],
                "gold": ["Earn 2 points per $1 spent", "Double points on birthday month", "Free standard shipping always", "Priority customer support", "Early sale access"],
                "platinum": ["Earn 3 points per $1 spent", "Triple points on birthday month", "Free express shipping always", "Dedicated account manager", "VIP events access", "Exclusive product launches"],
            }

            return json.dumps({
                "status": "success",
                "customer_id": customer_id,
                "customer_name": customer.get("name"),
                "loyalty": {
                    "current_tier": tier,
                    "points_balance": points,
                    "total_points_earned": total_earned,
                    "total_points_redeemed": total_redeemed,
                    "purchase_points_earned": purchase_points,
                    "bonus_points_earned": bonus_points,
                },
                "tier_progress": {
                    "current_tier": tier,
                    "next_tier": next_tier,
                    "points_to_next_tier": points_to_next,
                    "progress_pct": progress_pct,
                    "current_tier_benefits": tier_benefits.get(tier, []),
                    "next_tier_benefits": tier_benefits.get(next_tier, []) if next_tier else [],
                },
                "available_rewards": available,
                "rewards_almost_reachable": unavailable_close,
                "recent_events": events_sorted[:10],
            }, indent=2)

        except Exception as e:
            logger.error(f"get_loyalty_summary error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def process_points_transaction(
        customer_id: str,
        points: int,
        transaction_type: str,
        reason: str,
        reference_id: Optional[str] = None,
    ) -> str:
        """
        Award or deduct loyalty points for a customer.
        transaction_type: 'earn' (award points), 'redeem' (deduct), 'adjust' (manual), 'expire'.
        points: number of points (always positive; type determines direction).
        reason: human-readable reason for the transaction.
        reference_id: optional linked order/campaign ID.
        Returns new balance and any tier change information.
        """
        try:
            customer = data_store.get_customer(customer_id)
            if not customer:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            if points <= 0:
                return json.dumps({"status": "error", "message": "Points must be a positive integer"})

            valid_types = {"earn", "redeem", "adjust", "expire"}
            if transaction_type not in valid_types:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid transaction_type. Must be one of: {', '.join(valid_types)}",
                })

            current_points = customer.get("points_balance", 0)
            old_tier = customer.get("tier", "bronze")

            # Determine point delta
            if transaction_type in ("earn", "adjust"):
                delta = points
            else:  # redeem or expire
                delta = -points
                if current_points + delta < 0:
                    return json.dumps({
                        "status": "error",
                        "message": f"Insufficient points. Balance: {current_points}, Requested: {points}",
                    })

            new_points = current_points + delta

            # Determine new tier based on points
            thresholds = data_store.get_tier_thresholds()
            new_tier = old_tier
            for tier_name, bounds in thresholds.items():
                min_pts = bounds["min"]
                max_pts = bounds["max"]
                if max_pts is None:
                    if new_points >= min_pts:
                        new_tier = tier_name
                elif min_pts <= new_points <= max_pts:
                    new_tier = tier_name
                    break

            tier_upgraded = new_tier != old_tier and (
                ["bronze", "silver", "gold", "platinum"].index(new_tier) >
                ["bronze", "silver", "gold", "platinum"].index(old_tier)
            )
            tier_downgraded = new_tier != old_tier and not tier_upgraded

            # Update customer
            updates = {"points_balance": new_points}
            if tier_upgraded or tier_downgraded:
                updates["tier"] = new_tier

            data_store.update_customer(customer_id, updates)

            # Record event
            event_type_map = {
                "earn": "points_earned",
                "redeem": "points_redeemed",
                "adjust": "points_adjusted",
                "expire": "points_expired",
            }
            event_id = f"LE{datetime.now().strftime('%Y%m%d%H%M%S')}"
            event = {
                "event_id": event_id,
                "customer_id": customer_id,
                "event_type": event_type_map[transaction_type],
                "date": datetime.now().isoformat()[:10],
                "points": delta,
                "reference_id": reference_id,
                "description": reason,
            }
            data_store.add_loyalty_event(event)

            return json.dumps({
                "status": "success",
                "customer_id": customer_id,
                "transaction": {
                    "type": transaction_type,
                    "points_delta": delta,
                    "reason": reason,
                    "reference_id": reference_id,
                    "event_id": event_id,
                },
                "balance": {
                    "previous_points": current_points,
                    "new_points": new_points,
                    "previous_tier": old_tier,
                    "new_tier": new_tier,
                },
                "tier_change": {
                    "occurred": tier_upgraded or tier_downgraded,
                    "type": "upgrade" if tier_upgraded else "downgrade" if tier_downgraded else "none",
                    "from_tier": old_tier,
                    "to_tier": new_tier,
                    "message": (
                        f"Congratulations! Upgraded to {new_tier.upper()}!" if tier_upgraded
                        else f"Tier adjusted to {new_tier}" if tier_downgraded
                        else "Tier unchanged"
                    ),
                },
            }, indent=2)

        except Exception as e:
            logger.error(f"process_points_transaction error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def get_tier_upgrade_candidates(
        from_tier: str,
        within_points: int = 500,
    ) -> str:
        """
        Find customers who are close to upgrading their loyalty tier.
        from_tier: current tier to analyze ('bronze', 'silver', 'gold').
        within_points: max points gap from next tier (default 500 = within 500 points).
        Returns customers sorted by points gap with CLV and engagement data.
        """
        try:
            valid_tiers = ["bronze", "silver", "gold"]
            if from_tier.lower() not in valid_tiers:
                return json.dumps({
                    "status": "error",
                    "message": f"from_tier must be one of: {', '.join(valid_tiers)}",
                })

            thresholds = data_store.get_tier_thresholds()
            next_tier = data_store.get_next_tier(from_tier.lower())
            if not next_tier:
                return json.dumps({
                    "status": "success",
                    "message": f"{from_tier.capitalize()} is already the highest tier",
                    "candidates": [],
                })

            next_tier_min = thresholds[next_tier]["min"]
            customers = data_store.get_customers()
            tier_customers = [c for c in customers if c.get("tier", "").lower() == from_tier.lower()]

            candidates = []
            for customer in tier_customers:
                points = customer.get("points_balance", 0)
                points_needed = next_tier_min - points

                if 0 < points_needed <= within_points:
                    # Get last purchase date
                    txns = data_store.get_customer_transactions(customer["customer_id"], limit=1)
                    last_purchase = txns[0]["date"] if txns else "N/A"
                    days_since = (datetime.now() - datetime.fromisoformat(last_purchase)).days if txns else 999

                    candidates.append({
                        "customer_id": customer["customer_id"],
                        "name": customer["name"],
                        "email": customer["email"],
                        "current_tier": from_tier,
                        "next_tier": next_tier,
                        "current_points": points,
                        "points_needed": points_needed,
                        "pct_of_way": round((points - thresholds[from_tier.lower()]["min"]) / (next_tier_min - thresholds[from_tier.lower()]["min"]) * 100, 1),
                        "lifetime_value": customer.get("lifetime_value", 0),
                        "last_purchase_date": last_purchase,
                        "days_since_purchase": days_since,
                        "preferred_channel": customer.get("preferred_channel", "email"),
                    })

            candidates.sort(key=lambda x: x["points_needed"])

            total_clv = round(sum(c["lifetime_value"] for c in candidates), 2)
            avg_points_needed = round(sum(c["points_needed"] for c in candidates) / max(len(candidates), 1), 0)

            return json.dumps({
                "status": "success",
                "from_tier": from_tier,
                "next_tier": next_tier,
                "next_tier_threshold": next_tier_min,
                "within_points_filter": within_points,
                "candidate_count": len(candidates),
                "total_clv_of_candidates": total_clv,
                "avg_points_needed": avg_points_needed,
                "candidates": candidates[:20],
                "recommended_campaign": {
                    "type": "loyalty",
                    "offer": f"Bonus points to reach {next_tier.capitalize()}",
                    "suggested_message": f"You're so close to {next_tier.capitalize()} status! Earn just {avg_points_needed:.0f} more points to unlock exclusive benefits.",
                    "urgency": "high" if avg_points_needed <= 200 else "medium",
                },
            }, indent=2)

        except Exception as e:
            logger.error(f"get_tier_upgrade_candidates error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

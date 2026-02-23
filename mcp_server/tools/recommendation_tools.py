"""
Recommendation MCP Tools.
Provides product recommendations, similar customers, next-best-action,
trending products, and content recommendations.

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
    """Register all recommendation tools onto the FastMCP server instance."""

    @mcp.tool
    def get_product_recommendations(
        customer_id: str,
        n: int = 5,
        exclude_purchased: bool = True,
        category_filter: Optional[str] = None,
    ) -> str:
        """
        Get top N personalized product recommendations for a customer.
        Uses collaborative filtering based on purchase history and similar customers.
        customer_id: the customer to generate recommendations for.
        n: number of recommendations (default 5, max 10).
        exclude_purchased: if True, exclude products already bought.
        category_filter: restrict to a specific product category.
        Returns ranked recommendations with relevance scores and explanations.
        """
        try:
            n = min(n, 10)
            customer = data_store.get_customer(customer_id)
            if not customer:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            # Get customer's purchase history
            customer_txns = data_store.get_customer_transactions(customer_id)
            purchased_ids = data_store.get_customer_product_ids(customer_id)
            purchased_categories = [t["category"] for t in customer_txns]
            category_freq: dict[str, int] = {}
            for cat in purchased_categories:
                category_freq[cat] = category_freq.get(cat, 0) + 1

            # Get all products
            products = data_store.get_products()
            if category_filter:
                products = [p for p in products if p["category"] == category_filter]

            # Find similar customers (shared category preferences)
            all_customers = data_store.get_customers()
            similar_customer_products: dict[str, int] = {}
            customer_cats = set(category_freq.keys())

            for other in all_customers:
                if other["customer_id"] == customer_id:
                    continue
                other_txns = data_store.get_customer_transactions(other["customer_id"])
                other_cats = {t["category"] for t in other_txns}
                overlap = len(customer_cats & other_cats)
                if overlap >= 1:
                    for txn in other_txns:
                        pid = txn["product_id"]
                        similar_customer_products[pid] = similar_customer_products.get(pid, 0) + overlap

            # Score products
            scored = []
            for product in products:
                pid = product["product_id"]

                if exclude_purchased and pid in purchased_ids:
                    continue

                # Score components
                # 1. Category affinity (does customer buy from this category?)
                cat_score = category_freq.get(product["category"], 0) * 3

                # 2. Collaborative: similar customers bought this
                collab_score = similar_customer_products.get(pid, 0) * 0.5

                # 3. Product quality (rating)
                rating_score = (product.get("avg_rating", 3) - 3) * 2

                # 4. Popularity
                popularity_score = min(product.get("purchase_count", 0) / 100, 5)

                # 5. Margin bonus (slight preference for high-margin items)
                margin_score = product.get("margin_pct", 0) * 2

                total_score = cat_score + collab_score + rating_score + popularity_score + margin_score

                # Determine recommendation reason
                reasons = []
                if cat_score > 0:
                    reasons.append(f"matches your {product['category']} purchases")
                if collab_score > 0:
                    reasons.append("popular with customers like you")
                if product.get("avg_rating", 0) >= 4.5:
                    reasons.append(f"highly rated ({product['avg_rating']}★)")
                if not reasons:
                    reasons.append("trending in your region")

                scored.append({
                    "product_id": pid,
                    "name": product["name"],
                    "category": product["category"],
                    "subcategory": product.get("subcategory", ""),
                    "price": product["price"],
                    "avg_rating": product.get("avg_rating", 0),
                    "relevance_score": round(total_score, 3),
                    "reason": "; ".join(reasons[:2]),
                })

            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            top_n = scored[:n]

            return json.dumps({
                "status": "success",
                "customer_id": customer_id,
                "customer_name": customer.get("name"),
                "top_categories": list(category_freq.keys())[:3],
                "recommendations": top_n,
                "total_candidates": len(scored),
            }, indent=2)

        except Exception as e:
            logger.error(f"get_product_recommendations error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def get_similar_customers(
        customer_id: str,
        n: int = 5,
    ) -> str:
        """
        Find customers with similar purchase behavior to the reference customer.
        Uses Jaccard similarity on category preferences and product overlap.
        customer_id: reference customer.
        n: number of similar customers to return (max 10).
        Returns similar customers with similarity scores and shared interests.
        """
        try:
            n = min(n, 10)
            reference = data_store.get_customer(customer_id)
            if not reference:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            ref_products = data_store.get_customer_product_ids(customer_id)
            ref_txns = data_store.get_customer_transactions(customer_id)
            ref_cats = {t["category"] for t in ref_txns}

            if not ref_products:
                return json.dumps({
                    "status": "success",
                    "customer_id": customer_id,
                    "message": "No purchase history found for this customer",
                    "similar_customers": [],
                })

            all_customers = data_store.get_customers()
            similarities = []

            for other in all_customers:
                oid = other["customer_id"]
                if oid == customer_id:
                    continue

                other_products = data_store.get_customer_product_ids(oid)
                other_txns = data_store.get_customer_transactions(oid)
                other_cats = {t["category"] for t in other_txns}

                if not other_products:
                    continue

                # Jaccard similarity on products
                intersection = ref_products & other_products
                union = ref_products | other_products
                product_sim = len(intersection) / len(union) if union else 0

                # Category overlap
                cat_intersection = ref_cats & other_cats
                cat_sim = len(cat_intersection) / len(ref_cats | other_cats) if (ref_cats | other_cats) else 0

                # Combined similarity
                sim_score = round(product_sim * 0.6 + cat_sim * 0.4, 3)

                if sim_score > 0:
                    similarities.append({
                        "customer_id": oid,
                        "name": other.get("name"),
                        "tier": other.get("tier"),
                        "lifetime_value": other.get("lifetime_value", 0),
                        "similarity_score": sim_score,
                        "shared_categories": list(cat_intersection),
                        "shared_products": list(intersection)[:3],
                    })

            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

            return json.dumps({
                "status": "success",
                "reference_customer": {
                    "customer_id": customer_id,
                    "name": reference.get("name"),
                    "tier": reference.get("tier"),
                    "categories": list(ref_cats),
                },
                "similar_customers": similarities[:n],
            }, indent=2)

        except Exception as e:
            logger.error(f"get_similar_customers error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def get_next_best_action(
        customer_id: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Determine the recommended next action for a customer.
        Considers purchase recency, loyalty points, tier status, and engagement.
        customer_id: the target customer.
        context: optional context string (e.g., 'browsed electronics', 'loyalty tier due').
        Returns prioritized action list with expected value and triggers.
        """
        try:
            customer = data_store.get_customer(customer_id)
            if not customer:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            last_purchase_str = customer.get("last_purchase_date", "2020-01-01")
            try:
                last_purchase = datetime.fromisoformat(last_purchase_str)
                days_inactive = (datetime.now() - last_purchase).days
            except ValueError:
                days_inactive = 999

            tier = customer.get("tier", "bronze")
            points = customer.get("points_balance", 0)
            clv = customer.get("lifetime_value", 0)
            channel = customer.get("preferred_channel", "email")

            # Get loyalty events for context
            loyalty_events = data_store.get_loyalty_events(customer_id)
            recent_events = sorted(loyalty_events, key=lambda x: x.get("date", ""), reverse=True)[:5]

            # Tier thresholds
            thresholds = data_store.get_tier_thresholds()
            next_tier = data_store.get_next_tier(tier)
            points_to_upgrade = None
            if next_tier and next_tier in thresholds:
                points_to_upgrade = max(0, thresholds[next_tier]["min"] - points)

            actions = []

            # Action 1: Reactivation (if dormant)
            if days_inactive > 60:
                priority = "critical" if days_inactive > 120 else "high"
                actions.append({
                    "action_type": "reactivation",
                    "priority": priority,
                    "priority_score": 10 if priority == "critical" else 8,
                    "description": f"Re-engage via {channel} with personalized offer",
                    "trigger": f"{days_inactive} days since last purchase",
                    "expected_value": round(clv * 0.15, 2),
                    "recommended_offer": "25-30% discount on favorite category",
                })

            # Action 2: Tier upgrade push
            if points_to_upgrade is not None and 0 < points_to_upgrade <= 1500:
                actions.append({
                    "action_type": "tier_upgrade_nudge",
                    "priority": "high",
                    "priority_score": 9,
                    "description": f"Only {points_to_upgrade} points away from {next_tier.upper()} tier",
                    "trigger": f"Near {next_tier} tier threshold",
                    "expected_value": round(clv * 0.25, 2),
                    "recommended_offer": f"Bonus points on next purchase to reach {next_tier}",
                })

            # Action 3: Points redemption (high balance)
            if points >= 2000:
                actions.append({
                    "action_type": "points_redemption",
                    "priority": "medium",
                    "priority_score": 6,
                    "description": f"Encourage redemption of {points:,} accumulated points",
                    "trigger": "High points balance may expire or reduce engagement",
                    "expected_value": round(points * 0.01, 2),
                    "recommended_offer": "Reminder email with top reward redemption options",
                })

            # Action 4: Cross-sell (if recently active)
            if days_inactive <= 30:
                recent_txns = data_store.get_customer_transactions(customer_id, limit=3)
                recent_categories = list({t["category"] for t in recent_txns})
                actions.append({
                    "action_type": "cross_sell",
                    "priority": "medium",
                    "priority_score": 7,
                    "description": f"Cross-sell complementary products in {', '.join(recent_categories[:2])}",
                    "trigger": "Recent purchase creates upsell opportunity window",
                    "expected_value": round(clv * 0.08, 2),
                    "recommended_offer": "Personalized 'Complete the look' or 'You might also like' email",
                })

            # Action 5: Context-based action
            if context:
                context_lower = context.lower()
                if "electronics" in context_lower:
                    actions.append({
                        "action_type": "category_retargeting",
                        "priority": "high",
                        "priority_score": 8,
                        "description": "Retarget with electronics category offer",
                        "trigger": "Browsed electronics category",
                        "expected_value": round(clv * 0.12, 2),
                        "recommended_offer": "10% off electronics + free shipping",
                    })

            # Sort by priority score
            actions.sort(key=lambda x: x["priority_score"], reverse=True)

            if not actions:
                actions.append({
                    "action_type": "engagement",
                    "priority": "low",
                    "priority_score": 3,
                    "description": "Send curated newsletter with trending products",
                    "trigger": "Regular engagement maintenance",
                    "expected_value": round(clv * 0.03, 2),
                    "recommended_offer": "Monthly curated picks based on preferences",
                })

            return json.dumps({
                "status": "success",
                "customer_id": customer_id,
                "customer_name": customer.get("name"),
                "tier": tier,
                "days_since_purchase": days_inactive,
                "points_balance": points,
                "points_to_next_tier": points_to_upgrade,
                "context": context,
                "actions": actions[:5],
                "primary_action": actions[0] if actions else None,
            }, indent=2)

        except Exception as e:
            logger.error(f"get_next_best_action error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def get_trending_products(
        category: Optional[str] = None,
        window_days: int = 30,
        n: int = 10,
    ) -> str:
        """
        Get trending products based on purchase velocity in the recent window.
        category: optional category filter (None = all categories).
        window_days: lookback window for trend analysis (default 30 days).
        n: number of top trending products to return (max 15).
        Returns products ranked by growth rate with purchase velocity metrics.
        """
        try:
            n = min(n, 15)
            now = datetime.now()
            current_window_start = (now - timedelta(days=window_days)).isoformat()[:10]
            previous_window_start = (now - timedelta(days=window_days * 2)).isoformat()[:10]

            # Current period purchases
            current_txns = data_store.get_transactions(date_from=current_window_start)
            # Previous period purchases (for comparison)
            prev_txns = data_store.get_transactions(
                date_from=previous_window_start,
                date_to=current_window_start,
            )

            if category:
                current_txns = [t for t in current_txns if t["category"] == category]
                prev_txns = [t for t in prev_txns if t["category"] == category]

            # Count purchases per product per period
            current_counts: dict[str, int] = {}
            for t in current_txns:
                pid = t["product_id"]
                current_counts[pid] = current_counts.get(pid, 0) + t.get("quantity", 1)

            prev_counts: dict[str, int] = {}
            for t in prev_txns:
                pid = t["product_id"]
                prev_counts[pid] = prev_counts.get(pid, 0) + t.get("quantity", 1)

            products = data_store.get_products()
            product_map = {p["product_id"]: p for p in products}

            trending = []
            for pid, count in current_counts.items():
                prev_count = prev_counts.get(pid, 0)
                growth_pct = round(
                    ((count - prev_count) / prev_count * 100) if prev_count > 0 else 100.0,
                    1,
                )
                product = product_map.get(pid, {})
                if not product:
                    continue
                if category and product.get("category") != category:
                    continue

                trending.append({
                    "product_id": pid,
                    "name": product.get("name", "Unknown"),
                    "category": product.get("category"),
                    "price": product.get("price", 0),
                    "avg_rating": product.get("avg_rating", 0),
                    "current_period_sales": count,
                    "previous_period_sales": prev_count,
                    "growth_pct": growth_pct,
                    "trend": "rising" if growth_pct > 10 else "stable" if growth_pct > -10 else "declining",
                })

            # Sort by growth then by absolute volume
            trending.sort(key=lambda x: (x["growth_pct"], x["current_period_sales"]), reverse=True)

            # Category breakdown
            cat_volumes: dict[str, int] = {}
            for t in current_txns:
                cat = t["category"]
                cat_volumes[cat] = cat_volumes.get(cat, 0) + t.get("quantity", 1)
            top_categories = sorted(cat_volumes.items(), key=lambda x: x[1], reverse=True)[:5]

            return json.dumps({
                "status": "success",
                "window_days": window_days,
                "category_filter": category,
                "trending_products": trending[:n],
                "total_products_tracked": len(trending),
                "top_categories_by_volume": [
                    {"category": cat, "units_sold": units}
                    for cat, units in top_categories
                ],
                "period": {
                    "current": f"{current_window_start} to {now.isoformat()[:10]}",
                    "previous": f"{previous_window_start} to {current_window_start}",
                },
            }, indent=2)

        except Exception as e:
            logger.error(f"get_trending_products error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool
    def get_content_recommendations(
        customer_id: str,
        channel: str = "email",
    ) -> str:
        """
        Generate content and messaging topic recommendations for a customer.
        Based on purchase history, preferences, and loyalty tier.
        customer_id: target customer.
        channel: delivery channel ('email', 'sms', 'push').
        Returns content topics with relevance scores and example subject lines.
        """
        try:
            customer = data_store.get_customer(customer_id)
            if not customer:
                return json.dumps({"status": "error", "message": f"Customer {customer_id} not found"})

            # Analyze purchase patterns
            txns = data_store.get_customer_transactions(customer_id)
            category_freq: dict[str, int] = {}
            total_spend: dict[str, float] = {}
            for t in txns:
                cat = t["category"]
                category_freq[cat] = category_freq.get(cat, 0) + 1
                total_spend[cat] = total_spend.get(cat, 0) + float(t.get("amount", 0))

            top_cats = sorted(category_freq.items(), key=lambda x: x[1], reverse=True)[:3]

            tier = customer.get("tier", "bronze")
            points = customer.get("points_balance", 0)

            # Build content topics
            topics = []

            # Category-based content
            for cat, freq in top_cats:
                relevance = min(freq / max(sum(category_freq.values()), 1) + 0.5, 1.0)
                topics.append({
                    "topic": f"{cat} New Arrivals & Deals",
                    "relevance": round(relevance, 2),
                    "rationale": f"Customer's top category ({freq} purchases)",
                    "example_subject_lines": _generate_subject_lines(cat, tier, channel),
                })

            # Loyalty/points content
            if points >= 1000:
                topics.append({
                    "topic": "Loyalty Rewards & Points Redemption",
                    "relevance": 0.85,
                    "rationale": f"Has {points:,} redeemable points",
                    "example_subject_lines": [
                        f"You have {points:,} points waiting to be redeemed!",
                        f"Don't let your {tier.upper()} rewards expire",
                        "Exclusive rewards just for you",
                    ],
                })

            # Seasonal/trending content
            topics.append({
                "topic": "Trending Products This Week",
                "relevance": 0.65,
                "rationale": "Generic trending content for engagement maintenance",
                "example_subject_lines": [
                    "🔥 What's hot this week",
                    "Top picks: Most popular right now",
                    "Don't miss these trending items",
                ],
            })

            # VIP content for high-tier customers
            if tier in ("gold", "platinum"):
                topics.append({
                    "topic": f"{tier.capitalize()} Member Exclusive Access",
                    "relevance": 0.90,
                    "rationale": f"High-value {tier} tier customer deserves VIP treatment",
                    "example_subject_lines": [
                        f"Exclusive: Early access for {tier.upper()} members only",
                        "You're invited: Private sale starts now",
                        f"As a {tier.capitalize()} member, you get first pick",
                    ],
                })

            topics.sort(key=lambda x: x["relevance"], reverse=True)

            # Channel-specific adjustments
            channel_notes = {
                "email": "Full content with images and multiple CTAs",
                "sms": "Short and direct (max 160 chars). Focus on single offer.",
                "push": "Single compelling headline + emoji. 50-char limit.",
            }

            return json.dumps({
                "status": "success",
                "customer_id": customer_id,
                "customer_name": customer.get("name"),
                "tier": tier,
                "channel": channel,
                "channel_note": channel_notes.get(channel, "Standard content"),
                "topics": topics[:4],
                "optimal_send_time": _get_optimal_send_time(customer),
            }, indent=2)

        except Exception as e:
            logger.error(f"get_content_recommendations error: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    def _generate_subject_lines(category: str, tier: str, channel: str) -> list[str]:
        tier_perk = "VIP " if tier in ("gold", "platinum") else ""
        if channel == "sms":
            return [f"{tier_perk}Sale: {category} up to 20% off. Shop now."]
        lines = [
            f"New in {category}: Handpicked just for you",
            f"{tier_perk}Exclusive {category} deals this week",
            f"Your {category} wishlist is waiting",
        ]
        return lines

    def _get_optimal_send_time(customer: dict) -> str:
        channel = customer.get("preferred_channel", "email")
        city = customer.get("city", "")
        eastern_cities = {"New York", "Boston", "Philadelphia", "Atlanta", "Miami", "Charlotte"}
        western_cities = {"San Francisco", "Los Angeles", "Seattle", "San Jose", "San Diego", "Las Vegas"}

        if city in eastern_cities:
            tz = "ET"
            morning = "9:00 AM"
            evening = "7:00 PM"
        elif city in western_cities:
            tz = "PT"
            morning = "9:00 AM"
            evening = "7:00 PM"
        else:
            tz = "CT"
            morning = "9:00 AM"
            evening = "7:00 PM"

        if channel == "email":
            return f"Tuesday/Thursday {morning} {tz}"
        elif channel == "sms":
            return f"Weekday {morning} {tz}"
        else:  # push
            return f"Weekday {evening} {tz}"

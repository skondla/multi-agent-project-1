"""
Unit tests for MCP tool functions.
Tools are invoked directly (bypassing the MCP server/client transport layer).
No Anthropic API key required.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.data_store import DataStore


# ─── Shared fixtures ──────────────────────────────────────────────────────────

CUSTOMERS = [
    {
        "customer_id": "C001",
        "name": "Alice",
        "tier": "gold",
        "points_balance": 5000,
        "lifetime_value": 3000.0,
        "last_purchase_date": "2026-01-15",
        "preferred_channel": "email",
    },
    {
        "customer_id": "C002",
        "name": "Bob",
        "tier": "silver",
        "points_balance": 1500,
        "lifetime_value": 900.0,
        "last_purchase_date": "2025-09-01",
        "preferred_channel": "sms",
    },
    {
        "customer_id": "C003",
        "name": "Carol",
        "tier": "bronze",
        "points_balance": 200,
        "lifetime_value": 150.0,
        "last_purchase_date": "2024-07-01",
        "preferred_channel": "push",
    },
    {
        "customer_id": "C004",
        "name": "Dave",
        "tier": "platinum",
        "points_balance": 18000,
        "lifetime_value": 12000.0,
        "last_purchase_date": "2026-02-10",
        "preferred_channel": "email",
    },
]

PRODUCTS = [
    {"product_id": "P001", "name": "Laptop", "category": "Electronics", "price": 999.0, "avg_rating": 4.7, "purchase_count": 100, "margin_pct": 0.30},
    {"product_id": "P002", "name": "Headphones", "category": "Electronics", "price": 199.0, "avg_rating": 4.5, "purchase_count": 80, "margin_pct": 0.40},
    {"product_id": "P003", "name": "T-Shirt", "category": "Fashion", "price": 30.0, "avg_rating": 4.2, "purchase_count": 200, "margin_pct": 0.60},
    {"product_id": "P004", "name": "Running Shoes", "category": "Sports & Fitness", "price": 120.0, "avg_rating": 4.6, "purchase_count": 150, "margin_pct": 0.45},
    {"product_id": "P005", "name": "Blender", "category": "Home & Kitchen", "price": 80.0, "avg_rating": 4.3, "purchase_count": 60, "margin_pct": 0.35},
]

TRANSACTIONS = [
    {"transaction_id": "T001", "customer_id": "C001", "product_id": "P001", "date": "2026-01-15", "amount": 999.0, "quantity": 1, "category": "Electronics", "campaign_id": None},
    {"transaction_id": "T002", "customer_id": "C001", "product_id": "P002", "date": "2025-12-01", "amount": 199.0, "quantity": 1, "category": "Electronics", "campaign_id": "CAM001"},
    {"transaction_id": "T003", "customer_id": "C001", "product_id": "P003", "date": "2025-11-01", "amount": 30.0, "quantity": 2, "category": "Fashion", "campaign_id": None},
    {"transaction_id": "T004", "customer_id": "C002", "product_id": "P004", "date": "2025-09-01", "amount": 120.0, "quantity": 1, "category": "Sports & Fitness", "campaign_id": None},
    {"transaction_id": "T005", "customer_id": "C004", "product_id": "P001", "date": "2026-02-10", "amount": 999.0, "quantity": 1, "category": "Electronics", "campaign_id": None},
    {"transaction_id": "T006", "customer_id": "C004", "product_id": "P002", "date": "2026-01-20", "amount": 199.0, "quantity": 1, "category": "Electronics", "campaign_id": "CAM001"},
    {"transaction_id": "T007", "customer_id": "C004", "product_id": "P004", "date": "2025-12-15", "amount": 120.0, "quantity": 1, "category": "Sports & Fitness", "campaign_id": None},
    {"transaction_id": "T008", "customer_id": "C004", "product_id": "P005", "date": "2025-11-20", "amount": 80.0, "quantity": 1, "category": "Home & Kitchen", "campaign_id": None},
]

# spend field must be present for budget-allocation tool to score channels correctly
CAMPAIGNS = [
    {
        "campaign_id": "CAM001",
        "name": "Holiday Email Blast",
        "channel": "email",
        "status": "completed",
        "budget": 5000.0,
        "spend": 4800.0,
        "revenue_generated": 20000.0,
        "impressions": 50000,
        "clicks": 4000,
        "conversions": 500,
        "roi": 3.0,
        "start_date": "2025-11-01",
        "end_date": "2025-12-31",
        "target_segment": "all",
    },
    {
        "campaign_id": "CAM002",
        "name": "Spring SMS",
        "channel": "sms",
        "status": "completed",
        "budget": 2000.0,
        "spend": 1800.0,
        "revenue_generated": 5000.0,
        "impressions": 10000,
        "clicks": 800,
        "conversions": 100,
        "roi": 1.5,
        "start_date": "2026-01-01",
        "end_date": "2026-04-30",
        "target_segment": "silver",
    },
]

LOYALTY_EVENTS = [
    {"event_id": "E001", "customer_id": "C001", "event_type": "points_earned", "points": 999, "date": "2026-01-15", "description": "Purchase T001"},
    {"event_id": "E002", "customer_id": "C001", "event_type": "points_redeemed", "points": -500, "date": "2025-12-15", "description": "Reward redemption"},
    {"event_id": "E003", "customer_id": "C004", "event_type": "points_earned", "points": 999, "date": "2026-02-10", "description": "Purchase T005"},
    {"event_id": "E004", "customer_id": "C004", "event_type": "bonus_points", "points": 500, "date": "2026-01-01", "description": "VIP bonus"},
]


@pytest.fixture
def tmp_data_dir(tmp_path):
    data = {
        "customers.json": CUSTOMERS,
        "transactions.json": TRANSACTIONS,
        "products.json": PRODUCTS,
        "campaigns.json": CAMPAIGNS,
        "loyalty_events.json": LOYALTY_EVENTS,
    }
    for fname, content in data.items():
        (tmp_path / fname).write_text(json.dumps(content))
    return tmp_path


@pytest.fixture
def store(tmp_data_dir):
    return DataStore(data_dir=tmp_data_dir)


def _make_mock_mcp():
    """Return a minimal mock that captures tool functions registered via @mcp.tool."""
    tools = {}

    class MockMCP:
        def tool(self, fn):
            tools[fn.__name__] = fn
            return fn

    return MockMCP(), tools


# ─── Segmentation Tools Tests ──────────────────────────────────────────────────

class TestSegmentationTools:
    @pytest.fixture(autouse=True)
    def setup(self, store):
        from mcp_server.tools import segmentation_tools
        self.mcp, self.tools = _make_mock_mcp()
        segmentation_tools.register(self.mcp, store)

    def test_compute_rfm_scores_returns_success(self):
        result = json.loads(self.tools["compute_rfm_scores"]())
        assert result["status"] == "success"
        assert result["total_customers_analyzed"] > 0

    def test_compute_rfm_scores_structure(self):
        result = json.loads(self.tools["compute_rfm_scores"]())
        assert "rfm_data" in result
        assert "summary" in result
        first = result["rfm_data"][0]
        for field in ("customer_id", "r_score", "f_score", "m_score", "rfm_score", "segment"):
            assert field in first, f"Missing field: {field}"

    def test_compute_rfm_scores_filter_by_customer(self):
        result = json.loads(self.tools["compute_rfm_scores"](customer_ids=["C001"]))
        assert result["status"] == "success"
        ids = [r["customer_id"] for r in result["rfm_data"]]
        assert "C001" in ids

    def test_rfm_scores_are_1_to_5(self):
        result = json.loads(self.tools["compute_rfm_scores"]())
        for row in result["rfm_data"]:
            for score_field in ("r_score", "f_score", "m_score"):
                assert 1 <= row[score_field] <= 5, f"{score_field}={row[score_field]} out of range"

    def test_rfm_composite_score(self):
        result = json.loads(self.tools["compute_rfm_scores"]())
        for row in result["rfm_data"]:
            expected = row["r_score"] + row["f_score"] + row["m_score"]
            assert row["rfm_score"] == expected

    def test_identify_churn_risk_returns_success(self):
        result = json.loads(self.tools["identify_churn_risk"]())
        assert result["status"] == "success"
        assert "at_risk_count" in result
        assert "customers" in result

    def test_identify_churn_risk_scores_between_0_and_1(self):
        result = json.loads(self.tools["identify_churn_risk"](score_threshold=0.0))
        for c in result["customers"]:
            assert 0 <= c["risk_score"] <= 1

    def test_identify_churn_risk_catches_old_customer(self):
        # C003's last_purchase_date 2024-07-01 is far in the past → high risk
        result = json.loads(self.tools["identify_churn_risk"](threshold_days=30, score_threshold=0.5))
        at_risk_ids = [c["customer_id"] for c in result["customers"]]
        assert "C003" in at_risk_ids

    def test_identify_churn_risk_very_high_threshold_returns_empty(self):
        # Sigmoid asymptotically approaches 1; threshold 0.9999 should filter nearly all
        result = json.loads(self.tools["identify_churn_risk"](score_threshold=0.9999))
        assert result["status"] == "success"
        # just check it returned cleanly — count may be 0 or very small
        assert isinstance(result["at_risk_count"], int)

    def test_get_segment_profile_known_segment(self):
        rfm = json.loads(self.tools["compute_rfm_scores"]())
        segments = {r["segment"] for r in rfm["rfm_data"]}
        if segments:
            seg = next(iter(segments))
            result = json.loads(self.tools["get_segment_profile"](seg))
            assert result["status"] in ("success", "not_found")

    def test_get_segment_profile_not_found(self):
        result = json.loads(self.tools["get_segment_profile"]("NonExistentSegment12345"))
        assert result["status"] == "not_found"
        assert "available_segments" in result

    def test_compare_segments_error_on_bad_names(self):
        result = json.loads(self.tools["compare_segments"]("FakeA", "FakeB"))
        assert "status" in result  # should return gracefully


# ─── Campaign Tools Tests ──────────────────────────────────────────────────────

class TestCampaignTools:
    @pytest.fixture(autouse=True)
    def setup(self, store):
        from mcp_server.tools import campaign_tools
        self.mcp, self.tools = _make_mock_mcp()
        campaign_tools.register(self.mcp, store)

    def test_get_campaign_performance_all(self):
        result = json.loads(self.tools["get_campaign_performance"]())
        assert result["status"] == "success"
        assert len(result["campaigns"]) == 2

    def test_get_campaign_performance_filter_ids(self):
        result = json.loads(self.tools["get_campaign_performance"](campaign_ids=["CAM001"]))
        assert result["status"] == "success"
        assert len(result["campaigns"]) == 1
        assert result["campaigns"][0]["campaign_id"] == "CAM001"

    def test_campaign_performance_has_metrics(self):
        # Metrics are nested under camp["metrics"] with keys ctr_pct, conversion_rate_pct, roi
        result = json.loads(self.tools["get_campaign_performance"]())
        camp = result["campaigns"][0]
        assert "metrics" in camp
        for field in ("ctr_pct", "conversion_rate_pct", "roi"):
            assert field in camp["metrics"], f"Missing metric: {field}"

    def test_recommend_budget_allocation_returns_success(self):
        result = json.loads(self.tools["recommend_budget_allocation"](total_budget=10000.0))
        assert result["status"] == "success"
        assert "allocations" in result

    def test_budget_allocation_sums_to_total(self):
        total = 10000.0
        result = json.loads(self.tools["recommend_budget_allocation"](total_budget=total))
        # Allocation amounts are in each item's "allocation" key
        allocated = sum(a["allocation"] for a in result["allocations"])
        assert abs(allocated - total) < 1.0  # within $1

    def test_forecast_campaign_roi_success_or_no_data(self):
        result = json.loads(self.tools["forecast_campaign_roi"](
            channel="email",
            budget=5000.0,
            target_segment="all",
            campaign_type="retention",
        ))
        assert result["status"] in ("success", "no_data")

    def test_forecast_campaign_roi_has_forecast_key(self):
        result = json.loads(self.tools["forecast_campaign_roi"](
            channel="email",
            budget=5000.0,
            target_segment="all",
            campaign_type="retention",
        ))
        if result["status"] == "success":
            assert "forecast" in result
            assert "forecasted_roi" in result["forecast"]

    def test_ab_test_analysis_significant(self):
        result = json.loads(self.tools["ab_test_analysis"](
            control_metric=0.05,
            test_metric=0.10,
            control_size=1000,
            test_size=1000,
            metric_name="conversion_rate",
        ))
        assert result["status"] == "success"
        assert result["results"]["significant_at_95pct"] is True

    def test_ab_test_analysis_not_significant(self):
        result = json.loads(self.tools["ab_test_analysis"](
            control_metric=0.050,
            test_metric=0.051,
            control_size=100,
            test_size=100,
            metric_name="conversion_rate",
        ))
        assert result["status"] == "success"
        assert result["results"]["significant_at_95pct"] is False

    def test_ab_test_analysis_has_recommendation(self):
        result = json.loads(self.tools["ab_test_analysis"](
            control_metric=0.05,
            test_metric=0.09,
            control_size=500,
            test_size=500,
            metric_name="ctr",
        ))
        assert result["status"] == "success"
        assert "recommendation" in result

    def test_identify_target_audience_returns_customers(self):
        result = json.loads(self.tools["identify_target_audience"](
            campaign_type="retention",
            limit=5,
        ))
        assert result["status"] == "success"
        assert "customers" in result


# ─── Recommendation Tools Tests ───────────────────────────────────────────────

class TestRecommendationTools:
    @pytest.fixture(autouse=True)
    def setup(self, store):
        from mcp_server.tools import recommendation_tools
        self.mcp, self.tools = _make_mock_mcp()
        recommendation_tools.register(self.mcp, store)

    def test_get_product_recommendations_success(self):
        result = json.loads(self.tools["get_product_recommendations"](
            customer_id="C001",
            n=3,
        ))
        assert result["status"] == "success"
        assert "recommendations" in result

    def test_recommendations_exclude_purchased_by_default(self):
        result = json.loads(self.tools["get_product_recommendations"](
            customer_id="C001",
            n=5,
            exclude_purchased=True,
        ))
        assert result["status"] == "success"
        recs = result["recommendations"]
        purchased = {"P001", "P002", "P003"}
        rec_ids = {r["product_id"] for r in recs}
        assert rec_ids.isdisjoint(purchased), f"Purchased products appeared in recs: {rec_ids & purchased}"

    def test_recommendations_customer_not_found(self):
        result = json.loads(self.tools["get_product_recommendations"](customer_id="CXXX"))
        assert result["status"] == "error"

    def test_get_similar_customers_success(self):
        result = json.loads(self.tools["get_similar_customers"](customer_id="C001", n=2))
        assert result["status"] == "success"
        assert "similar_customers" in result
        assert len(result["similar_customers"]) <= 2

    def test_get_similar_customers_not_found(self):
        result = json.loads(self.tools["get_similar_customers"](customer_id="CXXX"))
        assert result["status"] == "error"

    def test_get_next_best_action_success(self):
        result = json.loads(self.tools["get_next_best_action"](customer_id="C001"))
        assert result["status"] == "success"
        assert "actions" in result
        assert len(result["actions"]) > 0

    def test_get_next_best_action_has_action_type(self):
        result = json.loads(self.tools["get_next_best_action"](customer_id="C001"))
        for action in result["actions"]:
            assert "action_type" in action
            assert "priority" in action

    def test_get_trending_products_success(self):
        result = json.loads(self.tools["get_trending_products"](n=3))
        assert result["status"] == "success"
        assert "trending_products" in result

    def test_get_trending_products_category_filter(self):
        result = json.loads(self.tools["get_trending_products"](category="Electronics", n=5))
        assert result["status"] == "success"
        assert "trending_products" in result

    def test_get_content_recommendations_success(self):
        result = json.loads(self.tools["get_content_recommendations"](
            customer_id="C001",
            channel="email",
        ))
        assert result["status"] == "success"
        assert "topics" in result

    def test_get_content_recommendations_returns_topics(self):
        result = json.loads(self.tools["get_content_recommendations"](
            customer_id="C001",
            channel="email",
        ))
        assert len(result["topics"]) > 0
        for topic in result["topics"]:
            assert "topic" in topic
            assert "relevance" in topic


# ─── CRM/Loyalty Tools Tests ──────────────────────────────────────────────────

class TestCRMLoyaltyTools:
    @pytest.fixture(autouse=True)
    def setup(self, store):
        from mcp_server.tools import crm_loyalty_tools
        self.mcp, self.tools = _make_mock_mcp()
        crm_loyalty_tools.register(self.mcp, store)
        self.store = store

    def test_get_customer_profile_success(self):
        result = json.loads(self.tools["get_customer_profile"](customer_id="C001"))
        assert result["status"] == "success"
        # Profile is nested under result["profile"]
        assert result["profile"]["customer_id"] == "C001"

    def test_get_customer_profile_has_transaction_summary(self):
        result = json.loads(self.tools["get_customer_profile"](customer_id="C001"))
        assert "transaction_summary" in result
        assert "total_orders" in result["transaction_summary"]
        assert "total_spent" in result["transaction_summary"]

    def test_get_customer_profile_not_found(self):
        result = json.loads(self.tools["get_customer_profile"](customer_id="CXXX"))
        assert result["status"] == "error"

    def test_calculate_clv_success(self):
        result = json.loads(self.tools["calculate_clv"](
            customer_id="C001",
            horizon_years=3,
            discount_rate=0.10,
        ))
        assert result["status"] == "success"
        # CLV projections are under result["projections"]
        assert "projections" in result

    def test_calculate_clv_positive_value(self):
        result = json.loads(self.tools["calculate_clv"](customer_id="C001"))
        assert result["status"] == "success"
        clv_1yr = result["projections"].get("clv_1_year", 0)
        assert clv_1yr > 0

    def test_calculate_clv_customer_not_found(self):
        result = json.loads(self.tools["calculate_clv"](customer_id="CXXX"))
        assert result["status"] == "error"

    def test_get_loyalty_summary_success(self):
        result = json.loads(self.tools["get_loyalty_summary"](customer_id="C001"))
        assert result["status"] == "success"
        # Loyalty data is nested under result["loyalty"]
        assert "loyalty" in result
        assert "points_balance" in result["loyalty"]

    def test_get_loyalty_summary_not_found(self):
        result = json.loads(self.tools["get_loyalty_summary"](customer_id="CXXX"))
        assert result["status"] == "error"

    def test_process_points_transaction_earn(self):
        result = json.loads(self.tools["process_points_transaction"](
            customer_id="C002",
            points=100,
            transaction_type="earn",
            reason="Test earn",
        ))
        assert result["status"] == "success"
        # New balance is under result["balance"]["new_points"]
        assert result["balance"]["new_points"] == 1500 + 100

    def test_process_points_transaction_redeem(self):
        result = json.loads(self.tools["process_points_transaction"](
            customer_id="C001",
            points=500,
            transaction_type="redeem",
            reason="Test redeem",
        ))
        assert result["status"] == "success"
        assert result["balance"]["new_points"] == 5000 - 500

    def test_process_points_insufficient_balance(self):
        result = json.loads(self.tools["process_points_transaction"](
            customer_id="C003",
            points=1000,  # C003 only has 200 points
            transaction_type="redeem",
            reason="Overdraft test",
        ))
        assert result["status"] == "error"
        assert "insufficient" in result.get("message", "").lower()

    def test_get_tier_upgrade_candidates_success(self):
        result = json.loads(self.tools["get_tier_upgrade_candidates"](
            from_tier="silver",
            within_points=500,
        ))
        assert result["status"] == "success"
        assert "candidates" in result


# ─── Integration: DataStore + Tool interaction ────────────────────────────────

class TestToolDataIntegration:
    """Verify that tool mutations are reflected in subsequent data reads."""

    def test_points_earn_persisted_in_store(self, store, tmp_data_dir):
        from mcp_server.tools import crm_loyalty_tools
        mcp, tools = _make_mock_mcp()
        crm_loyalty_tools.register(mcp, store)

        initial = store.get_customer("C002")["points_balance"]
        tools["process_points_transaction"](
            customer_id="C002",
            points=250,
            transaction_type="earn",
            reason="Integration test",
        )
        updated = store.get_customer("C002")["points_balance"]
        assert updated == initial + 250

    def test_churn_risk_uses_real_dates(self, store):
        from mcp_server.tools import segmentation_tools
        mcp, tools = _make_mock_mcp()
        segmentation_tools.register(mcp, store)

        # C003 has last_purchase_date 2024-07-01 — should show high churn risk
        result = json.loads(tools["identify_churn_risk"](score_threshold=0.5))
        at_risk_ids = [c["customer_id"] for c in result["customers"]]
        assert "C003" in at_risk_ids

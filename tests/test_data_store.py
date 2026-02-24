"""
Unit tests for DataStore — covers data loading, filtering, and mutation.
No Anthropic API key required.
"""
import json
import tempfile
import shutil
from pathlib import Path

import pytest

from mcp_server.data_store import DataStore


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_CUSTOMERS = [
    {
        "customer_id": "C001",
        "name": "Alice Test",
        "email": "alice@test.com",
        "tier": "gold",
        "points_balance": 4000,
        "lifetime_value": 2500.00,
        "last_purchase_date": "2025-12-01",
        "preferred_channel": "email",
    },
    {
        "customer_id": "C002",
        "name": "Bob Test",
        "email": "bob@test.com",
        "tier": "silver",
        "points_balance": 1200,
        "lifetime_value": 800.00,
        "last_purchase_date": "2025-06-15",
        "preferred_channel": "sms",
    },
    {
        "customer_id": "C003",
        "name": "Carol Test",
        "email": "carol@test.com",
        "tier": "bronze",
        "points_balance": 300,
        "lifetime_value": 150.00,
        "last_purchase_date": "2024-09-10",
        "preferred_channel": "push",
    },
]

SAMPLE_TRANSACTIONS = [
    {
        "transaction_id": "T001",
        "customer_id": "C001",
        "product_id": "P001",
        "date": "2025-12-01",
        "amount": 120.00,
        "quantity": 1,
        "category": "Electronics",
        "campaign_id": None,
    },
    {
        "transaction_id": "T002",
        "customer_id": "C001",
        "product_id": "P002",
        "date": "2025-11-15",
        "amount": 45.00,
        "quantity": 2,
        "category": "Fashion",
        "campaign_id": "CAM001",
    },
    {
        "transaction_id": "T003",
        "customer_id": "C002",
        "product_id": "P003",
        "date": "2025-06-15",
        "amount": 200.00,
        "quantity": 1,
        "category": "Electronics",
        "campaign_id": None,
    },
]

SAMPLE_PRODUCTS = [
    {
        "product_id": "P001",
        "name": "Test Widget",
        "category": "Electronics",
        "price": 120.00,
        "avg_rating": 4.5,
        "purchase_count": 50,
        "margin_pct": 0.35,
    },
    {
        "product_id": "P002",
        "name": "Test Shirt",
        "category": "Fashion",
        "price": 45.00,
        "avg_rating": 4.2,
        "purchase_count": 30,
        "margin_pct": 0.50,
    },
    {
        "product_id": "P003",
        "name": "Test Gadget",
        "category": "Electronics",
        "price": 200.00,
        "avg_rating": 4.8,
        "purchase_count": 20,
        "margin_pct": 0.40,
    },
]

SAMPLE_CAMPAIGNS = [
    {
        "campaign_id": "CAM001",
        "name": "Test Email Campaign",
        "channel": "email",
        "status": "completed",
        "budget": 5000.00,
        "revenue_generated": 15000.00,
        "impressions": 10000,
        "clicks": 800,
        "conversions": 120,
        "roi": 2.00,
        "start_date": "2025-10-01",
        "end_date": "2025-10-31",
    },
    {
        "campaign_id": "CAM002",
        "name": "Active SMS Campaign",
        "channel": "sms",
        "status": "active",
        "budget": 2000.00,
        "revenue_generated": 4000.00,
        "impressions": 5000,
        "clicks": 300,
        "conversions": 40,
        "roi": 1.00,
        "start_date": "2026-01-01",
        "end_date": "2026-03-31",
    },
]

SAMPLE_LOYALTY_EVENTS = [
    {
        "event_id": "EVT001",
        "customer_id": "C001",
        "type": "points_earned",
        "points": 120,
        "date": "2025-12-01",
        "description": "Purchase T001",
    },
    {
        "event_id": "EVT002",
        "customer_id": "C001",
        "type": "points_redeemed",
        "points": -500,
        "date": "2025-11-01",
        "description": "Redeemed for reward",
    },
    {
        "event_id": "EVT003",
        "customer_id": "C002",
        "type": "points_earned",
        "points": 200,
        "date": "2025-06-15",
        "description": "Purchase T003",
    },
]


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with sample JSON files."""
    files = {
        "customers.json": SAMPLE_CUSTOMERS,
        "transactions.json": SAMPLE_TRANSACTIONS,
        "products.json": SAMPLE_PRODUCTS,
        "campaigns.json": SAMPLE_CAMPAIGNS,
        "loyalty_events.json": SAMPLE_LOYALTY_EVENTS,
    }
    for filename, data in files.items():
        (tmp_path / filename).write_text(json.dumps(data))
    return tmp_path


@pytest.fixture
def store(tmp_data_dir):
    """DataStore backed by temporary sample data."""
    return DataStore(data_dir=tmp_data_dir)


# ─── DataStore Load Tests ────────────────────────────────────────────────────

class TestDataStoreLoad:
    def test_loads_all_customers(self, store):
        customers = store.get_customers()
        assert len(customers) == 3

    def test_loads_all_transactions(self, store):
        txns = store.get_transactions()
        assert len(txns) == 3

    def test_loads_all_products(self, store):
        products = store.get_products()
        assert len(products) == 3

    def test_loads_all_campaigns(self, store):
        campaigns = store.get_campaigns()
        assert len(campaigns) == 2

    def test_loads_all_loyalty_events(self, store):
        events = store.get_loyalty_events()
        assert len(events) == 3

    def test_missing_file_does_not_raise(self, tmp_path):
        """DataStore should not crash if a file is missing."""
        # Only write customers.json — others are absent
        (tmp_path / "customers.json").write_text(json.dumps(SAMPLE_CUSTOMERS))
        ds = DataStore(data_dir=tmp_path)
        assert len(ds.get_customers()) == 3
        assert ds.get_transactions() == []


# ─── Customer Access Tests ────────────────────────────────────────────────────

class TestCustomerAccess:
    def test_get_all_customers(self, store):
        result = store.get_customers()
        ids = [c["customer_id"] for c in result]
        assert "C001" in ids
        assert "C002" in ids
        assert "C003" in ids

    def test_get_customers_by_ids(self, store):
        result = store.get_customers(ids=["C001", "C003"])
        assert len(result) == 2
        ids = {c["customer_id"] for c in result}
        assert ids == {"C001", "C003"}

    def test_get_customer_by_id(self, store):
        c = store.get_customer("C001")
        assert c is not None
        assert c["name"] == "Alice Test"
        assert c["tier"] == "gold"

    def test_get_customer_not_found(self, store):
        assert store.get_customer("NONEXISTENT") is None

    def test_update_customer(self, store, tmp_data_dir):
        updated = store.update_customer("C002", {"tier": "gold", "points_balance": 5500})
        assert updated is True
        c = store.get_customer("C002")
        assert c["tier"] == "gold"
        assert c["points_balance"] == 5500
        # Verify persisted to file
        saved = json.loads((tmp_data_dir / "customers.json").read_text())
        c002 = next(x for x in saved if x["customer_id"] == "C002")
        assert c002["tier"] == "gold"

    def test_update_nonexistent_customer(self, store):
        result = store.update_customer("NOPE", {"tier": "platinum"})
        assert result is False


# ─── Transaction Access Tests ─────────────────────────────────────────────────

class TestTransactionAccess:
    def test_get_all_transactions(self, store):
        assert len(store.get_transactions()) == 3

    def test_filter_by_customer_ids(self, store):
        txns = store.get_transactions(customer_ids=["C001"])
        assert len(txns) == 2
        assert all(t["customer_id"] == "C001" for t in txns)

    def test_filter_by_date_from(self, store):
        txns = store.get_transactions(date_from="2025-12-01")
        assert len(txns) == 1
        assert txns[0]["transaction_id"] == "T001"

    def test_filter_by_date_to(self, store):
        txns = store.get_transactions(date_to="2025-06-15")
        assert all(t["date"] <= "2025-06-15" for t in txns)

    def test_filter_by_category(self, store):
        txns = store.get_transactions(category="Electronics")
        assert len(txns) == 2
        assert all(t["category"] == "Electronics" for t in txns)

    def test_get_customer_transactions_sorted(self, store):
        txns = store.get_customer_transactions("C001")
        assert len(txns) == 2
        # Should be sorted descending by date
        assert txns[0]["date"] >= txns[1]["date"]

    def test_get_customer_transactions_limit(self, store):
        txns = store.get_customer_transactions("C001", limit=1)
        assert len(txns) == 1


# ─── Product Access Tests ──────────────────────────────────────────────────────

class TestProductAccess:
    def test_get_all_products(self, store):
        assert len(store.get_products()) == 3

    def test_get_products_by_ids(self, store):
        result = store.get_products(ids=["P001"])
        assert len(result) == 1
        assert result[0]["name"] == "Test Widget"

    def test_get_product_by_id(self, store):
        p = store.get_product("P002")
        assert p is not None
        assert p["category"] == "Fashion"

    def test_get_product_not_found(self, store):
        assert store.get_product("PXXX") is None

    def test_get_products_by_category(self, store):
        electronics = store.get_products_by_category("Electronics")
        assert len(electronics) == 2
        assert all(p["category"] == "Electronics" for p in electronics)

    def test_get_all_categories(self, store):
        cats = store.get_all_categories()
        assert "Electronics" in cats
        assert "Fashion" in cats
        assert cats == sorted(cats)  # should be sorted


# ─── Campaign Access Tests ─────────────────────────────────────────────────────

class TestCampaignAccess:
    def test_get_all_campaigns(self, store):
        assert len(store.get_campaigns()) == 2

    def test_filter_by_status(self, store):
        active = store.get_campaigns(status="active")
        assert len(active) == 1
        assert active[0]["campaign_id"] == "CAM002"

    def test_filter_completed(self, store):
        completed = store.get_campaigns(status="completed")
        assert len(completed) == 1
        assert completed[0]["campaign_id"] == "CAM001"

    def test_get_campaign_by_id(self, store):
        c = store.get_campaign("CAM001")
        assert c is not None
        assert c["channel"] == "email"

    def test_get_campaign_not_found(self, store):
        assert store.get_campaign("CAMXXX") is None

    def test_get_all_channels(self, store):
        channels = store.get_all_channels()
        assert "email" in channels
        assert "sms" in channels
        assert channels == sorted(channels)


# ─── Loyalty Event Tests ────────────────────────────────────────────────────────

class TestLoyaltyEvents:
    def test_get_all_loyalty_events(self, store):
        assert len(store.get_loyalty_events()) == 3

    def test_filter_by_customer(self, store):
        events = store.get_loyalty_events(customer_id="C001")
        assert len(events) == 2
        assert all(e["customer_id"] == "C001" for e in events)

    def test_add_loyalty_event(self, store, tmp_data_dir):
        new_event = {
            "event_id": "EVT004",
            "customer_id": "C003",
            "type": "bonus_points",
            "points": 100,
            "date": "2026-01-01",
            "description": "Welcome bonus",
        }
        store.add_loyalty_event(new_event)
        events = store.get_loyalty_events(customer_id="C003")
        assert len(events) == 1
        assert events[0]["event_id"] == "EVT004"
        # Verify persisted
        saved = json.loads((tmp_data_dir / "loyalty_events.json").read_text())
        assert any(e["event_id"] == "EVT004" for e in saved)


# ─── Computed Helpers Tests ────────────────────────────────────────────────────

class TestComputedHelpers:
    def test_get_product_purchase_counts(self, store):
        counts = store.get_product_purchase_counts()
        assert counts["P001"] == 1
        assert counts["P002"] == 2   # quantity=2 in T002
        assert counts["P003"] == 1

    def test_get_product_purchase_counts_since_date(self, store):
        counts = store.get_product_purchase_counts(since_date="2025-11-01")
        # T001 (2025-12-01) and T002 (2025-11-15) qualify; T003 (2025-06-15) does not
        assert "P001" in counts
        assert "P002" in counts
        assert "P003" not in counts

    def test_get_customer_product_ids(self, store):
        product_ids = store.get_customer_product_ids("C001")
        assert product_ids == {"P001", "P002"}

    def test_get_tier_thresholds(self, store):
        thresholds = store.get_tier_thresholds()
        assert thresholds["bronze"]["min"] == 0
        assert thresholds["platinum"]["max"] is None
        assert thresholds["silver"]["min"] == 1000

    def test_get_next_tier(self, store):
        assert store.get_next_tier("bronze") == "silver"
        assert store.get_next_tier("silver") == "gold"
        assert store.get_next_tier("gold") == "platinum"
        assert store.get_next_tier("platinum") is None

    def test_get_next_tier_case_insensitive(self, store):
        assert store.get_next_tier("BRONZE") == "silver"
        assert store.get_next_tier("Gold") == "platinum"

    def test_get_next_tier_unknown(self, store):
        assert store.get_next_tier("unknown_tier") is None

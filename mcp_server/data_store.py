"""
In-memory data store backed by JSON files.
Provides typed access methods and optional write-back for mutations.
CRITICAL: This module is imported by the MCP server process.
Never use print() - log to stderr only.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure logging goes to stderr only (never stdout - corrupts JSON-RPC)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


class DataStore:
    """
    Singleton-pattern data store that loads all JSON files at initialization.
    Provides typed get/update methods for each data domain.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = data_dir or DATA_DIR
        self._customers: list[dict] = []
        self._transactions: list[dict] = []
        self._products: list[dict] = []
        self._campaigns: list[dict] = []
        self._loyalty_events: list[dict] = []
        self._load_all()

    def _load_all(self) -> None:
        """Load all JSON data files into memory."""
        file_map = [
            ("_customers", "customers.json"),
            ("_transactions", "transactions.json"),
            ("_products", "products.json"),
            ("_campaigns", "campaigns.json"),
            ("_loyalty_events", "loyalty_events.json"),
        ]
        for attr, filename in file_map:
            path = self._data_dir / filename
            if path.exists():
                data = json.loads(path.read_text())
                setattr(self, attr, data)
                logger.info(f"Loaded {len(data)} records from {filename}")
            else:
                logger.warning(f"Data file not found: {path}")

    def _save(self, filename: str, data: list) -> None:
        """Write data back to JSON file."""
        path = self._data_dir / filename
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved {len(data)} records to {filename}")

    # ─── Customer Access ────────────────────────────────────────────────────

    def get_customers(self, ids: Optional[list[str]] = None) -> list[dict]:
        if ids is None:
            return self._customers
        id_set = set(ids)
        return [c for c in self._customers if c["customer_id"] in id_set]

    def get_customer(self, customer_id: str) -> Optional[dict]:
        for c in self._customers:
            if c["customer_id"] == customer_id:
                return c
        return None

    def update_customer(self, customer_id: str, updates: dict) -> bool:
        for i, c in enumerate(self._customers):
            if c["customer_id"] == customer_id:
                self._customers[i].update(updates)
                self._save("customers.json", self._customers)
                return True
        return False

    # ─── Transaction Access ─────────────────────────────────────────────────

    def get_transactions(
        self,
        customer_ids: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[dict]:
        txns = self._transactions
        if customer_ids:
            id_set = set(customer_ids)
            txns = [t for t in txns if t["customer_id"] in id_set]
        if date_from:
            txns = [t for t in txns if t["date"] >= date_from]
        if date_to:
            txns = [t for t in txns if t["date"] <= date_to]
        if category:
            txns = [t for t in txns if t["category"] == category]
        return txns

    def get_customer_transactions(
        self,
        customer_id: str,
        limit: Optional[int] = None,
    ) -> list[dict]:
        txns = [t for t in self._transactions if t["customer_id"] == customer_id]
        txns.sort(key=lambda x: x["date"], reverse=True)
        return txns[:limit] if limit else txns

    # ─── Product Access ─────────────────────────────────────────────────────

    def get_products(self, ids: Optional[list[str]] = None) -> list[dict]:
        if ids is None:
            return self._products
        id_set = set(ids)
        return [p for p in self._products if p["product_id"] in id_set]

    def get_product(self, product_id: str) -> Optional[dict]:
        for p in self._products:
            if p["product_id"] == product_id:
                return p
        return None

    def get_products_by_category(self, category: str) -> list[dict]:
        return [p for p in self._products if p["category"] == category]

    # ─── Campaign Access ─────────────────────────────────────────────────────

    def get_campaigns(
        self,
        ids: Optional[list[str]] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        camps = self._campaigns
        if ids:
            id_set = set(ids)
            camps = [c for c in camps if c["campaign_id"] in id_set]
        if status:
            camps = [c for c in camps if c.get("status") == status]
        return camps

    def get_campaign(self, campaign_id: str) -> Optional[dict]:
        for c in self._campaigns:
            if c["campaign_id"] == campaign_id:
                return c
        return None

    def add_campaign(self, campaign: dict) -> None:
        self._campaigns.append(campaign)
        self._save("campaigns.json", self._campaigns)

    # ─── Loyalty Event Access ────────────────────────────────────────────────

    def get_loyalty_events(
        self,
        customer_id: Optional[str] = None,
    ) -> list[dict]:
        if customer_id is None:
            return self._loyalty_events
        return [e for e in self._loyalty_events if e["customer_id"] == customer_id]

    def add_loyalty_event(self, event: dict) -> None:
        self._loyalty_events.append(event)
        self._save("loyalty_events.json", self._loyalty_events)

    # ─── Computed Helpers ───────────────────────────────────────────────────

    def get_all_categories(self) -> list[str]:
        """Get unique product categories."""
        return sorted({p["category"] for p in self._products})

    def get_all_channels(self) -> list[str]:
        """Get unique campaign channels from historical campaigns."""
        return sorted({c["channel"] for c in self._campaigns})

    def get_product_purchase_counts(self, since_date: Optional[str] = None) -> dict:
        """Return {product_id: purchase_count} for the given date range."""
        txns = self._transactions
        if since_date:
            txns = [t for t in txns if t["date"] >= since_date]
        counts: dict[str, int] = {}
        for t in txns:
            pid = t["product_id"]
            counts[pid] = counts.get(pid, 0) + t.get("quantity", 1)
        return counts

    def get_customer_product_ids(self, customer_id: str) -> set[str]:
        """Return set of product IDs purchased by a customer."""
        return {t["product_id"] for t in self._transactions if t["customer_id"] == customer_id}

    def get_tier_thresholds(self) -> dict:
        """Return loyalty tier point thresholds."""
        return {
            "bronze": {"min": 0, "max": 999},
            "silver": {"min": 1000, "max": 4999},
            "gold": {"min": 5000, "max": 14999},
            "platinum": {"min": 15000, "max": None},
        }

    def get_next_tier(self, current_tier: str) -> Optional[str]:
        """Get the tier above the current one."""
        order = ["bronze", "silver", "gold", "platinum"]
        try:
            idx = order.index(current_tier.lower())
            return order[idx + 1] if idx + 1 < len(order) else None
        except ValueError:
            return None

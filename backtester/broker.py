"""Order types and the simulated broker that fills them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import pandas as pd


@dataclass
class Order:
    symbol: str
    quantity: float  # positive = buy, negative = sell
    created_at: pd.Timestamp
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    @property
    def is_buy(self) -> bool:
        return self.quantity > 0


@dataclass
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float

    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class CostModel:
    """Transaction costs.

    commission_rate: fraction of traded value (0.001 = 10 bps)
    commission_min:  minimum commission per fill
    slippage_rate:   fraction the fill price moves against you (0.0005 = 5 bps)
    """

    commission_rate: float = 0.0
    commission_min: float = 0.0
    slippage_rate: float = 0.0

    def fill_price(self, price: float, is_buy: bool) -> float:
        return price * (1 + self.slippage_rate) if is_buy else price * (1 - self.slippage_rate)

    def commission(self, quantity: float, price: float) -> float:
        if quantity == 0:
            return 0.0
        return max(abs(quantity) * price * self.commission_rate, self.commission_min)


@dataclass
class Broker:
    """Holds pending orders and fills them against the next bar.

    Orders placed during bar t are executed during bar t+1, which prevents
    look-ahead bias: a strategy can never trade at a price it used to decide.
    """

    costs: CostModel = field(default_factory=CostModel)
    allow_short: bool = False
    pending: List[Order] = field(default_factory=list)

    def submit(self, order: Order) -> None:
        if order.quantity != 0:
            self.pending.append(order)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        self.pending = [o for o in self.pending if symbol is not None and o.symbol != symbol]

    def _trigger_price(self, order: Order, bar: Mapping[str, float]) -> Optional[float]:
        """Return the raw (pre-slippage) execution price, or None if not triggered."""
        o, h, l = bar["open"], bar["high"], bar["low"]
        if order.stop_price is not None:
            # Stop orders become market orders once the stop is touched.
            if order.is_buy and h >= order.stop_price:
                return max(o, order.stop_price)
            if not order.is_buy and l <= order.stop_price:
                return min(o, order.stop_price)
            return None
        if order.limit_price is not None:
            if order.is_buy and l <= order.limit_price:
                return min(o, order.limit_price)
            if not order.is_buy and h >= order.limit_price:
                return max(o, order.limit_price)
            return None
        return o

    def process(
        self,
        timestamp: pd.Timestamp,
        bars: Dict[str, Mapping[str, float]],
        cash: float,
        positions: Dict[str, float],
    ) -> List[Fill]:
        """Try to fill pending orders against this bar's prices.

        Unfilled limit/stop orders stay pending (good-till-cancelled). Buys are
        scaled down to the available cash; sells are capped at the current
        position unless shorting is allowed.
        """
        fills: List[Fill] = []
        still_pending: List[Order] = []
        positions = dict(positions)

        # Process sells first so their proceeds can fund buys on the same bar.
        for order in sorted(self.pending, key=lambda o: o.is_buy):
            bar = bars.get(order.symbol)
            if bar is None:
                still_pending.append(order)
                continue
            raw = self._trigger_price(order, bar)
            if raw is None:
                still_pending.append(order)
                continue

            price = self.costs.fill_price(raw, order.is_buy)
            qty = order.quantity
            held = positions.get(order.symbol, 0.0)

            if not order.is_buy and not self.allow_short:
                qty = -min(-qty, max(held, 0.0))
            if order.is_buy:
                # Covering a short is always allowed; any new long beyond it must be paid for.
                cover = min(qty, max(-held, 0.0))
                if qty > cover:
                    budget = cash - cover * price - self.costs.commission(cover, price)
                    qty = cover + min(qty - cover, self._affordable(budget, price))
                    while qty > cover and qty * price + self.costs.commission(qty, price) > cash:
                        qty -= 1

            if qty == 0:
                continue

            commission = self.costs.commission(qty, price)
            cash -= qty * price + commission
            positions[order.symbol] = held + qty
            fills.append(Fill(timestamp, order.symbol, qty, price, commission))

        self.pending = still_pending
        return fills

    def _affordable(self, cash: float, price: float) -> float:
        """Largest whole-share quantity whose cost plus commission fits in cash."""
        if cash <= 0 or price <= 0:
            return 0.0
        qty = float(int(cash / (price * (1 + self.costs.commission_rate))))
        while qty > 0 and qty * price + self.costs.commission(qty, price) > cash:
            qty -= 1
        return qty

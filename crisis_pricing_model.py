"""
Crisis-Adaptive Dynamic Pricing Engine
=======================================
ML-driven pricing strategy that detects market crises (recession, war,
supply-chain shocks) and re-optimises prices to protect margin while
preserving demand and brand equity.

Author : Yaroslav Dobrianskyi
Usage  : python3 crisis_pricing_model.py [--demo] [--export-report]

Architecture (5 modules)

  ┌───────────────┬─────────────────────────────────────────────────────────┐
  │    Module     │                         Purpose                         │
  ├───────────────┼─────────────────────────────────────────────────────────┤
  │               │ Weighted composite scoring across 5 macro signals       │
  │ Crisis        │ (volatility, demand, cost, sentiment, supply            │
  │ Detector      │ disruption) → classifies into NONE / MILD / MODERATE /  │
  │               │ SEVERE                                                  │
  ├───────────────┼─────────────────────────────────────────────────────────┤
  │ Elasticity    │ GradientBoostedRegressor that predicts demand response  │
  │ Model         │ to price changes, with crisis×category interaction      │
  │               │ features (CV R² = 0.98)                                 │
  ├───────────────┼─────────────────────────────────────────────────────────┤
  │               │ 12-cell matrix (4 crisis levels × 3 product categories) │
  │ Strategy      │  selecting the right strategy — from margin_optimize in │
  │ Matrix        │  normal times to ethical_cap / liquidation in severe    │
  │               │ crisis                                                  │
  ├───────────────┼─────────────────────────────────────────────────────────┤
  │ Constrained   │ scipy bounded optimization maximizing contribution      │
  │ Optimizer     │ margin under guardrails (margin floor, daily damping    │
  │               │ ±3%, ethical caps on essentials)                        │
  ├───────────────┼─────────────────────────────────────────────────────────┤
  │ Scenario      │ What-if engine comparing 5 pre-built scenarios: normal, │
  │ Simulator     │  mild recession, severe recession, war disruption,      │
  │               │ supply-chain shock                                      │
  └───────────────┴─────────────────────────────────────────────────────────┘

  Key strategies by crisis severity

  - Normal → pure margin optimization across all categories
  - Mild → cost passthrough on essentials, selective discounts on
  semi-essentials
  - Moderate → capped cost passthrough, demand protection, deep discounts on
  discretionary
  - Severe → ethical caps on essentials (max +10%), survival pricing,
  liquidation of discretionary inventory

  Results summary from the demo run

  ┌────────────────────┬────────────┬────────────────┐
  │      Scenario      │ Avg Margin │ Revenue Impact │
  ├────────────────────┼────────────┼────────────────┤
  │ Normal             │ 60.4%      │ +0.6%          │
  ├────────────────────┼────────────┼────────────────┤
  │ Mild Recession     │ 58.6%      │ -0.2%          │
  ├────────────────────┼────────────┼────────────────┤
  │ Severe Recession   │ 55.7%      │ -0.4%          │
  ├────────────────────┼────────────┼────────────────┤
  │ War Disruption     │ 44.2%      │ +4.3%          │
  ├────────────────────┼────────────┼────────────────┤
  │ Supply Chain Shock │ 38.8%      │ +2.6%          │
  └────────────────────┴────────────┴────────────────┘

 Usage

  source ~/Documents/Scripts/.venv/bin/activate

  # Full scenario comparison
  python3 ~/crisis_pricing_model.py --demo

  # Export to CSV
  python3 ~/crisis_pricing_model.py --demo --export-report
  results.csv

  # Single war-disruption scenario
  python3 ~/crisis_pricing_model.py
"""

import argparse
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# 1.  DOMAIN DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

class CrisisLevel(Enum):
    NONE = 0
    MILD = 1       # early signals: volatility spike, sentiment drop
    MODERATE = 2   # confirmed downturn: demand contraction, cost surge
    SEVERE = 3     # full crisis: war, recession, supply-chain collapse


class ProductCategory(Enum):
    ESSENTIAL = "essential"          # food, medicine, utilities
    SEMI_ESSENTIAL = "semi_essential"  # electronics, home goods
    DISCRETIONARY = "discretionary"  # luxury, travel, entertainment


@dataclass
class Product:
    sku: str
    name: str
    category: ProductCategory
    base_price: float
    unit_cost: float
    current_price: float = 0.0
    inventory_days: float = 30.0  # days of stock remaining

    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.base_price

    @property
    def base_margin(self) -> float:
        return (self.base_price - self.unit_cost) / self.base_price


@dataclass
class MarketSignals:
    """Real-time market context fed into the pricing engine."""
    volatility_index: float       # 0-100, e.g. VIX-like
    demand_change_pct: float      # vs. baseline, e.g. -0.25 = 25% drop
    cost_change_pct: float        # supply-side cost inflation
    competitor_price_change: float # avg competitor move, e.g. +0.10
    consumer_sentiment: float     # 0-100 index
    supply_disruption_score: float  # 0-1, 1 = total disruption


# ──────────────────────────────────────────────────────────────────────
# 2.  CRISIS DETECTOR — scores market state into CrisisLevel
# ──────────────────────────────────────────────────────────────────────

class CrisisDetector:
    """
    Weighted composite score across macro signals.
    In production this would be an ML classifier trained on historical
    crisis periods; here we use an interpretable rule-based scorer
    so the logic is auditable by business stakeholders.
    """

    WEIGHTS = {
        "volatility": 0.20,
        "demand": 0.25,
        "cost": 0.15,
        "sentiment": 0.20,
        "supply": 0.20,
    }
    THRESHOLDS = {
        CrisisLevel.MILD: 0.30,
        CrisisLevel.MODERATE: 0.55,
        CrisisLevel.SEVERE: 0.75,
    }

    def score(self, signals: MarketSignals) -> float:
        """Return composite crisis score in [0, 1]."""
        s = (
            self.WEIGHTS["volatility"] * min(signals.volatility_index / 80, 1.0)
            + self.WEIGHTS["demand"] * min(abs(signals.demand_change_pct) / 0.5, 1.0)
            + self.WEIGHTS["cost"] * min(signals.cost_change_pct / 0.5, 1.0)
            + self.WEIGHTS["sentiment"] * (1 - signals.consumer_sentiment / 100)
            + self.WEIGHTS["supply"] * signals.supply_disruption_score
        )
        return round(np.clip(s, 0, 1), 4)

    def detect(self, signals: MarketSignals) -> CrisisLevel:
        s = self.score(signals)
        if s >= self.THRESHOLDS[CrisisLevel.SEVERE]:
            return CrisisLevel.SEVERE
        if s >= self.THRESHOLDS[CrisisLevel.MODERATE]:
            return CrisisLevel.MODERATE
        if s >= self.THRESHOLDS[CrisisLevel.MILD]:
            return CrisisLevel.MILD
        return CrisisLevel.NONE


# ──────────────────────────────────────────────────────────────────────
# 3.  ELASTICITY MODEL — predicts demand response to price changes
# ──────────────────────────────────────────────────────────────────────

class ElasticityModel:
    """
    Gradient-boosted demand model.  Features include price ratio,
    crisis level, category, inventory pressure, and competitor delta.
    Trained on synthetic historical data for demo; in production
    this trains on actual transaction logs.
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.is_fitted = False

    # ── feature engineering ──────────────────────────────────────────

    @staticmethod
    def _encode_category(cat: ProductCategory) -> list[float]:
        return [
            1.0 if cat == ProductCategory.ESSENTIAL else 0.0,
            1.0 if cat == ProductCategory.SEMI_ESSENTIAL else 0.0,
            1.0 if cat == ProductCategory.DISCRETIONARY else 0.0,
        ]

    def _build_features(
        self,
        price_ratio: float,
        category: ProductCategory,
        crisis_level: int,
        inventory_days: float,
        competitor_delta: float,
    ) -> np.ndarray:
        cat_enc = self._encode_category(category)
        return np.array([
            price_ratio,
            price_ratio ** 2,           # non-linear price effect
            crisis_level,
            crisis_level * price_ratio,  # interaction: crisis amplifies sensitivity
            inventory_days / 90,
            competitor_delta,
            *cat_enc,
        ]).reshape(1, -1)

    # ── synthetic training data ──────────────────────────────────────

    def _generate_training_data(self, n: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate historical (price_change → demand_change) observations
        across normal and crisis periods with realistic elasticity curves.
        """
        rng = np.random.RandomState(42)

        rows, targets = [], []
        for _ in range(n):
            cat = rng.choice(list(ProductCategory))
            crisis = rng.choice([0, 1, 2, 3], p=[0.55, 0.20, 0.15, 0.10])
            price_ratio = rng.uniform(0.70, 1.40)  # 30% discount to 40% markup
            inv_days = rng.uniform(5, 120)
            comp_delta = rng.normal(0, 0.08)

            # ground-truth elasticity by category
            base_elast = {
                ProductCategory.ESSENTIAL: -0.4,
                ProductCategory.SEMI_ESSENTIAL: -1.2,
                ProductCategory.DISCRETIONARY: -2.0,
            }[cat]

            # crisis amplifies elasticity for non-essentials
            crisis_mult = 1.0 + 0.3 * crisis * (0.2 if cat == ProductCategory.ESSENTIAL else 1.0)
            elasticity = base_elast * crisis_mult

            # demand change = elasticity * ln(price_ratio) + noise
            demand_change = elasticity * np.log(price_ratio)
            demand_change += 0.15 * comp_delta  # competitor effect
            demand_change += rng.normal(0, 0.04)  # noise

            feat = self._build_features(price_ratio, cat, crisis, inv_days, comp_delta)
            rows.append(feat.flatten())
            targets.append(demand_change)

        return np.array(rows), np.array(targets)

    # ── train & evaluate ─────────────────────────────────────────────

    def train(self, verbose: bool = False):
        X, y = self._generate_training_data()
        tscv = TimeSeriesSplit(n_splits=3)
        mae_scores, r2_scores = [], []
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
            preds = self.model.predict(X[val_idx])
            mae_scores.append(mean_absolute_error(y[val_idx], preds))
            r2_scores.append(r2_score(y[val_idx], preds))

        # final fit on full data
        self.model.fit(X, y)
        self.is_fitted = True

        if verbose:
            print(f"  Elasticity model trained — CV MAE: {np.mean(mae_scores):.4f}, "
                  f"R²: {np.mean(r2_scores):.4f}")

    def predict_demand_change(
        self,
        price_ratio: float,
        category: ProductCategory,
        crisis_level: int,
        inventory_days: float = 30,
        competitor_delta: float = 0.0,
    ) -> float:
        """Predict % demand change for a given price ratio."""
        if not self.is_fitted:
            self.train()
        feat = self._build_features(price_ratio, category, crisis_level, inventory_days, competitor_delta)
        return float(self.model.predict(feat)[0])


# ──────────────────────────────────────────────────────────────────────
# 4.  PRICING STRATEGY ENGINE — crisis-aware optimisation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PricingConstraints:
    """Guardrails that prevent destructive pricing moves."""
    max_price_increase_pct: float = 0.15    # cap single-period hike
    max_price_decrease_pct: float = 0.25    # cap single-period drop
    min_margin_floor: float = 0.05          # never go below 5% margin
    essential_markup_cap: float = 0.10      # ethical cap on essentials in crisis
    daily_change_damper: float = 0.03       # max 3% move per day (smoothing)


@dataclass
class PricingResult:
    sku: str
    old_price: float
    new_price: float
    predicted_demand_change: float
    expected_margin: float
    expected_revenue_change: float
    strategy_applied: str
    crisis_level: str


class CrisisPricingEngine:
    """
    Core engine that combines crisis detection, elasticity prediction,
    and constrained margin optimisation.

    Strategy matrix (crisis_level × category):
    ┌─────────────┬────────────────┬──────────────────┬────────────────────┐
    │ Crisis      │ Essential      │ Semi-Essential   │ Discretionary      │
    ├─────────────┼────────────────┼──────────────────┼────────────────────┤
    │ None        │ Standard       │ Standard         │ Standard           │
    │ Mild        │ Hold / +cost   │ Selective disc.  │ Moderate discount  │
    │ Moderate    │ Cost pass-thru │ Demand-protect   │ Deep discount      │
    │ Severe      │ Ethical cap    │ Survival pricing │ Liquidation        │
    └─────────────┴────────────────┴──────────────────┴────────────────────┘
    """

    def __init__(
        self,
        elasticity_model: ElasticityModel,
        constraints: Optional[PricingConstraints] = None,
    ):
        self.detector = CrisisDetector()
        self.elasticity = elasticity_model
        self.constraints = constraints or PricingConstraints()

    # ── strategy selection ───────────────────────────────────────────

    def _select_strategy(
        self, crisis: CrisisLevel, category: ProductCategory
    ) -> str:
        matrix = {
            CrisisLevel.NONE: {
                ProductCategory.ESSENTIAL: "margin_optimize",
                ProductCategory.SEMI_ESSENTIAL: "margin_optimize",
                ProductCategory.DISCRETIONARY: "margin_optimize",
            },
            CrisisLevel.MILD: {
                ProductCategory.ESSENTIAL: "cost_passthrough",
                ProductCategory.SEMI_ESSENTIAL: "selective_discount",
                ProductCategory.DISCRETIONARY: "moderate_discount",
            },
            CrisisLevel.MODERATE: {
                ProductCategory.ESSENTIAL: "cost_passthrough_capped",
                ProductCategory.SEMI_ESSENTIAL: "demand_protect",
                ProductCategory.DISCRETIONARY: "deep_discount",
            },
            CrisisLevel.SEVERE: {
                ProductCategory.ESSENTIAL: "ethical_cap",
                ProductCategory.SEMI_ESSENTIAL: "survival_pricing",
                ProductCategory.DISCRETIONARY: "liquidation",
            },
        }
        return matrix[crisis][category]

    # ── objective function for optimisation ──────────────────────────

    def _contribution_margin(
        self,
        price: float,
        product: Product,
        crisis_level: int,
        signals: MarketSignals,
    ) -> float:
        """
        Predicted contribution margin = (price - cost) * demand(price).
        We maximise this subject to constraints.
        Negative sign because scipy minimises.
        """
        price_ratio = price / product.base_price
        demand_change = self.elasticity.predict_demand_change(
            price_ratio=price_ratio,
            category=product.category,
            crisis_level=crisis_level,
            inventory_days=product.inventory_days,
            competitor_delta=signals.competitor_price_change,
        )
        relative_demand = 1.0 + demand_change
        relative_demand = max(relative_demand, 0.01)  # floor

        unit_cost = product.unit_cost * (1 + signals.cost_change_pct)
        margin_per_unit = price - unit_cost
        return -(margin_per_unit * relative_demand)  # negative for minimisation

    # ── constrained price bounds ─────────────────────────────────────

    def _price_bounds(
        self, product: Product, strategy: str, signals: MarketSignals
    ) -> tuple[float, float]:
        c = self.constraints
        unit_cost = product.unit_cost * (1 + signals.cost_change_pct)
        min_price = unit_cost / (1 - c.min_margin_floor)  # margin floor

        lower = max(
            product.current_price * (1 - c.max_price_decrease_pct),
            min_price,
        )
        upper = product.current_price * (1 + c.max_price_increase_pct)

        # ethical cap on essentials during crisis
        if strategy in ("ethical_cap", "cost_passthrough_capped"):
            upper = min(upper, product.base_price * (1 + c.essential_markup_cap))

        # liquidation allows deeper cuts
        if strategy == "liquidation":
            lower = max(unit_cost * 1.01, product.current_price * 0.50)

        return lower, upper

    # ── main pricing logic ───────────────────────────────────────────

    def price_product(
        self, product: Product, signals: MarketSignals
    ) -> PricingResult:
        crisis = self.detector.detect(signals)
        strategy = self._select_strategy(crisis, product.category)
        lower, upper = self._price_bounds(product, strategy, signals)

        result = minimize_scalar(
            self._contribution_margin,
            bounds=(lower, upper),
            method="bounded",
            args=(product, crisis.value, signals),
        )
        optimal_price = round(result.x, 2)

        # dampen daily change
        max_move = product.current_price * self.constraints.daily_change_damper
        if abs(optimal_price - product.current_price) > max_move:
            direction = 1 if optimal_price > product.current_price else -1
            optimal_price = round(product.current_price + direction * max_move, 2)

        # final prediction at chosen price
        price_ratio = optimal_price / product.base_price
        demand_chg = self.elasticity.predict_demand_change(
            price_ratio, product.category, crisis.value,
            product.inventory_days, signals.competitor_price_change,
        )
        unit_cost = product.unit_cost * (1 + signals.cost_change_pct)
        new_margin = (optimal_price - unit_cost) / optimal_price
        revenue_change = (1 + demand_chg) * (optimal_price / product.current_price) - 1

        return PricingResult(
            sku=product.sku,
            old_price=product.current_price,
            new_price=optimal_price,
            predicted_demand_change=round(demand_chg, 4),
            expected_margin=round(new_margin, 4),
            expected_revenue_change=round(revenue_change, 4),
            strategy_applied=strategy,
            crisis_level=crisis.name,
        )

    def price_catalog(
        self, products: list[Product], signals: MarketSignals
    ) -> pd.DataFrame:
        results = [self.price_product(p, signals) for p in products]
        return pd.DataFrame([r.__dict__ for r in results])


# ──────────────────────────────────────────────────────────────────────
# 5.  SCENARIO SIMULATOR — compare strategies across crisis levels
# ──────────────────────────────────────────────────────────────────────

class ScenarioSimulator:
    """Run what-if analysis across multiple crisis scenarios."""

    SCENARIOS = {
        "normal": MarketSignals(
            volatility_index=15, demand_change_pct=0.0,
            cost_change_pct=0.0, competitor_price_change=0.0,
            consumer_sentiment=72, supply_disruption_score=0.05,
        ),
        "mild_recession": MarketSignals(
            volatility_index=35, demand_change_pct=-0.10,
            cost_change_pct=0.05, competitor_price_change=-0.03,
            consumer_sentiment=50, supply_disruption_score=0.15,
        ),
        "severe_recession": MarketSignals(
            volatility_index=55, demand_change_pct=-0.30,
            cost_change_pct=0.12, competitor_price_change=-0.08,
            consumer_sentiment=30, supply_disruption_score=0.35,
        ),
        "war_disruption": MarketSignals(
            volatility_index=75, demand_change_pct=-0.35,
            cost_change_pct=0.40, competitor_price_change=0.05,
            consumer_sentiment=20, supply_disruption_score=0.80,
        ),
        "supply_chain_shock": MarketSignals(
            volatility_index=50, demand_change_pct=-0.15,
            cost_change_pct=0.55, competitor_price_change=0.12,
            consumer_sentiment=38, supply_disruption_score=0.70,
        ),
    }

    def __init__(self, engine: CrisisPricingEngine):
        self.engine = engine

    def run(self, products: list[Product]) -> pd.DataFrame:
        all_results = []
        for scenario_name, signals in self.SCENARIOS.items():
            # reset prices to base for fair comparison
            for p in products:
                p.current_price = p.base_price
            df = self.engine.price_catalog(products, signals)
            df.insert(0, "scenario", scenario_name)
            crisis_score = self.engine.detector.score(signals)
            df.insert(1, "crisis_score", crisis_score)
            all_results.append(df)
        return pd.concat(all_results, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# 6.  SAMPLE CATALOG
# ──────────────────────────────────────────────────────────────────────

SAMPLE_PRODUCTS = [
    Product("ESS-001", "Organic Rice 1kg", ProductCategory.ESSENTIAL, 4.50, 2.80),
    Product("ESS-002", "Paracetamol 20-pack", ProductCategory.ESSENTIAL, 3.20, 1.10),
    Product("SEM-001", "Wireless Mouse", ProductCategory.SEMI_ESSENTIAL, 29.99, 12.50),
    Product("SEM-002", "LED Desk Lamp", ProductCategory.SEMI_ESSENTIAL, 45.00, 18.00),
    Product("DIS-001", "Premium Headphones", ProductCategory.DISCRETIONARY, 199.00, 65.00),
    Product("DIS-002", "Spa Gift Set", ProductCategory.DISCRETIONARY, 89.00, 28.00),
    Product("ESS-003", "Cooking Oil 1L", ProductCategory.ESSENTIAL, 5.80, 3.40),
    Product("SEM-003", "USB-C Hub", ProductCategory.SEMI_ESSENTIAL, 39.99, 15.00),
    Product("DIS-003", "Designer Candle", ProductCategory.DISCRETIONARY, 55.00, 12.00),
]


# ──────────────────────────────────────────────────────────────────────
# 7.  REPORT GENERATOR
# ──────────────────────────────────────────────────────────────────────

def print_report(sim_df: pd.DataFrame) -> None:
    """Pretty-print scenario comparison to terminal."""
    print("\n" + "=" * 90)
    print("  CRISIS-ADAPTIVE PRICING ENGINE — Scenario Analysis")
    print("=" * 90)

    for scenario in sim_df["scenario"].unique():
        sdf = sim_df[sim_df["scenario"] == scenario]
        crisis_score = sdf["crisis_score"].iloc[0]
        crisis_level = sdf["crisis_level"].iloc[0]

        print(f"\n{'─' * 90}")
        print(f"  Scenario: {scenario.upper().replace('_', ' ')}")
        print(f"  Crisis Score: {crisis_score:.2f}  |  Level: {crisis_level}")
        print(f"{'─' * 90}")
        print(f"  {'SKU':<10} {'Product':<22} {'Old':>7} {'New':>7} {'Chg%':>7} "
              f"{'Demand':>8} {'Margin':>8} {'Strategy':<24}")
        print(f"  {'':─<10} {'':─<22} {'':─>7} {'':─>7} {'':─>7} {'':─>8} {'':─>8} {'':─<24}")

        for _, r in sdf.iterrows():
            pct = (r["new_price"] / r["old_price"] - 1) * 100
            print(
                f"  {r['sku']:<10} {r['sku']:<22} "
                f"€{r['old_price']:>6.2f} €{r['new_price']:>6.2f} {pct:>+6.1f}% "
                f"{r['predicted_demand_change']:>+7.1%} "
                f"{r['expected_margin']:>7.1%} "
                f"{r['strategy_applied']:<24}"
            )

        # scenario summary
        avg_margin = sdf["expected_margin"].mean()
        avg_rev_chg = sdf["expected_revenue_change"].mean()
        print(f"\n  Avg margin: {avg_margin:.1%}  |  Avg revenue impact: {avg_rev_chg:+.1%}")

    print(f"\n{'=' * 90}\n")


def export_csv(sim_df: pd.DataFrame, path: str) -> None:
    sim_df.to_csv(path, index=False)
    print(f"  Exported to {path}")


# ──────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crisis-Adaptive Dynamic Pricing Engine")
    parser.add_argument("--demo", action="store_true", help="Run full scenario simulation")
    parser.add_argument("--export-report", type=str, default=None,
                        help="Export results to CSV path")
    args = parser.parse_args()

    print("\n  Initialising Crisis Pricing Engine...")

    # train elasticity model
    elasticity = ElasticityModel()
    elasticity.train(verbose=True)

    # build engine
    engine = CrisisPricingEngine(elasticity)

    if args.demo:
        print("  Running scenario simulation across 5 crisis levels...")
        simulator = ScenarioSimulator(engine)
        results = simulator.run(SAMPLE_PRODUCTS)
        print_report(results)

        if args.export_report:
            export_csv(results, args.export_report)
    else:
        # single scenario — war disruption as default showcase
        print("\n  Running single scenario: WAR DISRUPTION\n")
        signals = ScenarioSimulator.SCENARIOS["war_disruption"]
        crisis = engine.detector.detect(signals)
        score = engine.detector.score(signals)
        print(f"  Crisis detected: {crisis.name} (score={score:.2f})")

        df = engine.price_catalog(SAMPLE_PRODUCTS, signals)
        for _, r in df.iterrows():
            pct = (r["new_price"] / r["old_price"] - 1) * 100
            print(f"  {r['sku']}  €{r['old_price']:.2f} → €{r['new_price']:.2f} "
                  f"({pct:+.1f}%)  strategy={r['strategy_applied']}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()

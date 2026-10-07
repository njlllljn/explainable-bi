"""Synthetic business data for a reproducible Explainable BI demonstration."""

import numpy as np
import pandas as pd


FEATURES = [
    "marketing_spend", "customer_retention", "discount_rate",
    "competitor_price", "purchase_frequency", "customer_tenure",
    "complaint_count", "average_order_value", "engagement_score",
]


def make_business_data(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    marketing = rng.uniform(5_000, 100_000, rows)
    retention = rng.uniform(0.35, 0.98, rows)
    discount = rng.uniform(0.0, 0.4, rows)
    competitor = rng.uniform(60, 160, rows)
    frequency = rng.uniform(0.2, 12, rows)
    tenure = rng.integers(1, 72, rows)
    complaints = rng.poisson(1.3, rows).clip(0, 8)
    order_value = rng.uniform(15, 350, rows)
    engagement = rng.uniform(0, 100, rows)

    revenue = (
        35_000 + marketing * 1.28 + retention * 72_000
        - discount * 62_000 + competitor * 380 + frequency * 4_100
        + tenure * 240 - complaints * 2_800 + order_value * 95
        + engagement * 210 + rng.normal(0, 12_000, rows)
    ).clip(10_000)
    churn_logit = (
        1.8 - retention * 4.1 - frequency * 0.28 - tenure * 0.025
        + complaints * 0.53 - engagement * 0.023 + discount * 1.1
        + rng.normal(0, 0.55, rows)
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churned = rng.binomial(1, churn_probability)
    risk = ((churn_probability > 0.58) | ((complaints >= 4) & (retention < 0.6))).astype(int)

    return pd.DataFrame({
        "marketing_spend": marketing.round(2),
        "customer_retention": retention.round(3),
        "discount_rate": discount.round(3),
        "competitor_price": competitor.round(2),
        "purchase_frequency": frequency.round(2),
        "customer_tenure": tenure,
        "complaint_count": complaints,
        "average_order_value": order_value.round(2),
        "engagement_score": engagement.round(1),
        "revenue": revenue.round(2),
        "churned": churned,
        "high_risk": risk,
    })

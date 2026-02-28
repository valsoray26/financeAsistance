"""
Standalone finance report generator — reads CSV directly, no database required.

Reports:
    annual  — Full-year overview with charts, insights, tables (HTML + PDF)
    monthly — Single-month detailed breakdown (HTML + PDF)
    savings — Savings advice with potential cuts and actionable tips (HTML)

Usage:
    python3 finance_report.py --year 2024                              # Annual HTML+PDF
    python3 finance_report.py --year 2024 --month 6                    # June monthly
    python3 finance_report.py --year 2024 --report savings             # Savings advice
    python3 finance_report.py --year 2024 --format html                # HTML only
    python3 finance_report.py --year 2024 --output-dir /tmp            # Custom output dir
    python3 finance_report.py --csv data/other_file.csv --year 2024    # Custom CSV
"""

import argparse
import base64
import calendar
import io
import os
import sys
from datetime import datetime
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
from jinja2 import Template


# ── Default paths ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "data", "personal_finance_2024_1500_transactions.csv")
DEFAULT_REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

# ── Colours ───────────────────────────────────────────────────
STEEL_BLUE = "#2E86AB"
AMBER = "#F6AE2D"
DARK_TEXT = "#2C3E50"
GREEN = "#27AE60"
RED = "#E74C3C"
LIGHT_BG = "#F8F9FA"
WHITE = "#FFFFFF"

CHART_PALETTE = [
    "#2E86AB", "#F6AE2D", "#E74C3C", "#27AE60", "#8E44AD",
    "#E67E22", "#1ABC9C", "#C0392B", "#2980B9", "#D35400",
    "#16A085", "#F39C12",
]

# Categories excluded from "reduce spending" advice (essentials)
ESSENTIAL_CATEGORIES = {"Rent", "Utilities", "Medical"}


# ══════════════════════════════════════════════════════════════
# 1. DATA LAYER  (CSV-based, no database)
# ══════════════════════════════════════════════════════════════

def load_csv(csv_path):
    """Read the finance CSV and return a normalised DataFrame."""
    df = pd.read_csv(csv_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%y %H:%M")
    df = df.rename(columns={
        "DateTime": "datetime",
        "Description": "description",
        "Merchant": "merchant",
        "Category_Name": "category",
        "Type": "type",
        "Amount_EUR": "amount",
        "Payment_Method": "payment_method",
    })
    df["abs_amount"] = df["amount"].abs()
    df["date"] = df["datetime"].dt.date
    return df


def fetch_transactions(all_data, year, month=None):
    """Filter transactions for the given year and optional month."""
    mask = all_data["datetime"].dt.year == year
    if month:
        mask &= all_data["datetime"].dt.month == month
    df = all_data[mask].copy().sort_values("datetime").reset_index(drop=True)
    return df


def fetch_previous_period(all_data, year, month=None):
    """Load prior period for comparison (previous month or previous year)."""
    if month:
        if month == 1:
            return fetch_transactions(all_data, year - 1, 12)
        return fetch_transactions(all_data, year, month - 1)
    return fetch_transactions(all_data, year - 1)


# ══════════════════════════════════════════════════════════════
# 2. ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════

def compute_summary(df):
    income = df.loc[df["type"] == "Income", "abs_amount"].sum()
    expenses = df.loc[df["type"] == "Expense", "abs_amount"].sum()
    n_days = max((df["date"].max() - df["date"].min()).days, 1) if len(df) else 1
    return {
        "income": income,
        "expenses": expenses,
        "net": income - expenses,
        "count": len(df),
        "avg_daily_spend": expenses / n_days if n_days else 0,
    }


def compute_category_breakdown(df):
    exp = df[df["type"] == "Expense"].copy()
    grp = exp.groupby("category")["abs_amount"].agg(["sum", "count"]).reset_index()
    grp.columns = ["category", "total", "count"]
    grp = grp.sort_values("total", ascending=False)
    grp["pct"] = (grp["total"] / grp["total"].sum() * 100).round(1)
    return grp


def compute_merchant_breakdown(df):
    exp = df[df["type"] == "Expense"].copy()
    grp = exp.groupby("merchant")["abs_amount"].agg(["sum", "count"]).reset_index()
    grp.columns = ["merchant", "total", "count"]
    grp = grp.sort_values("total", ascending=False)
    grp["pct"] = (grp["total"] / grp["total"].sum() * 100).round(1)
    return grp


def compute_payment_method_breakdown(df):
    exp = df[df["type"] == "Expense"].copy()
    grp = exp.groupby("payment_method")["abs_amount"].agg(["sum", "count"]).reset_index()
    grp.columns = ["method", "total", "count"]
    grp = grp.sort_values("total", ascending=False)
    grp["pct"] = (grp["total"] / grp["total"].sum() * 100).round(1)
    return grp


def compute_daily_trend(df):
    exp = df[df["type"] == "Expense"].copy()
    daily = exp.groupby("date")["abs_amount"].sum().reset_index()
    daily.columns = ["date", "total"]
    daily = daily.sort_values("date")
    daily["ma7"] = daily["total"].rolling(7, min_periods=1).mean()
    return daily


def compute_monthly_trend(df):
    df2 = df.copy()
    df2["month"] = df2["datetime"].dt.month
    inc = df2[df2["type"] == "Income"].groupby("month")["abs_amount"].sum()
    exp = df2[df2["type"] == "Expense"].groupby("month")["abs_amount"].sum()
    months = range(1, 13)
    return pd.DataFrame({
        "month": list(months),
        "income": [inc.get(m, 0) for m in months],
        "expenses": [exp.get(m, 0) for m in months],
    })


def compute_weekday_pattern(df):
    exp = df[df["type"] == "Expense"].copy()
    exp["weekday"] = exp["datetime"].dt.dayofweek  # 0=Mon
    grp = exp.groupby("weekday")["abs_amount"].mean().reset_index()
    grp.columns = ["weekday", "avg_spend"]
    grp["day_name"] = grp["weekday"].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )
    return grp.sort_values("weekday")


# ══════════════════════════════════════════════════════════════
# 3. INSIGHTS ENGINE
# ══════════════════════════════════════════════════════════════

def generate_insights(df, prev_df, summary, cat_breakdown, merchant_breakdown, is_annual):
    insights = []

    # --- Anomaly detection: transactions > mean + 2*std per category ---
    exp = df[df["type"] == "Expense"].copy()
    for cat in exp["category"].unique():
        cat_txns = exp[exp["category"] == cat]["abs_amount"]
        if len(cat_txns) < 5:
            continue
        mean, std = cat_txns.mean(), cat_txns.std()
        threshold = mean + 2 * std
        outliers = cat_txns[cat_txns > threshold]
        if len(outliers) > 0:
            insights.append({
                "severity": "warning",
                "text": f"Anomaly in {cat}: {len(outliers)} transaction(s) exceeded "
                        f"\u20ac{threshold:.0f} (mean + 2\u00d7SD). "
                        f"Largest: \u20ac{outliers.max():.2f}.",
            })

    # --- Period-over-period category changes ---
    if len(prev_df) > 0:
        prev_exp = prev_df[prev_df["type"] == "Expense"]
        prev_cat = prev_exp.groupby("category")["abs_amount"].sum()
        curr_cat = exp.groupby("category")["abs_amount"].sum()
        for cat in curr_cat.index:
            curr_val = curr_cat[cat]
            prev_val = prev_cat.get(cat, 0)
            if prev_val > 0:
                change_pct = (curr_val - prev_val) / prev_val * 100
                if change_pct > 20:
                    period_label = "previous year" if is_annual else "previous month"
                    insights.append({
                        "severity": "warning",
                        "text": f"{cat} spending up {change_pct:.0f}% vs {period_label} "
                                f"(\u20ac{prev_val:,.0f} \u2192 \u20ac{curr_val:,.0f}).",
                    })
                elif change_pct < -20:
                    period_label = "previous year" if is_annual else "previous month"
                    insights.append({
                        "severity": "info",
                        "text": f"{cat} spending down {abs(change_pct):.0f}% vs {period_label} "
                                f"(\u20ac{prev_val:,.0f} \u2192 \u20ac{curr_val:,.0f}).",
                    })

    # --- Budget alerts ---
    if summary["net"] < 0:
        insights.append({
            "severity": "critical",
            "text": f"Deficit alert: Expenses exceed income by \u20ac{abs(summary['net']):,.2f}.",
        })

    if len(cat_breakdown) > 0:
        top_cat = cat_breakdown.iloc[0]
        if top_cat["pct"] > 30:
            insights.append({
                "severity": "warning",
                "text": f"High category concentration: {top_cat['category']} accounts for "
                        f"{top_cat['pct']:.1f}% of all expenses (\u20ac{top_cat['total']:,.0f}).",
            })

    if len(merchant_breakdown) > 0:
        top_m = merchant_breakdown.iloc[0]
        if top_m["pct"] > 25:
            insights.append({
                "severity": "warning",
                "text": f"Merchant dependency: {top_m['merchant']} accounts for "
                        f"{top_m['pct']:.1f}% of spending (\u20ac{top_m['total']:,.0f}).",
            })

    # --- Spending patterns ---
    weekday_data = compute_weekday_pattern(df)
    if len(weekday_data) > 0:
        peak_day = weekday_data.loc[weekday_data["avg_spend"].idxmax()]
        insights.append({
            "severity": "info",
            "text": f"Highest average spending day: {peak_day['day_name']} "
                    f"(\u20ac{peak_day['avg_spend']:.2f}/day).",
        })

        weekday_avg = weekday_data[weekday_data["weekday"] < 5]["avg_spend"].mean()
        weekend_avg = weekday_data[weekday_data["weekday"] >= 5]["avg_spend"].mean()
        if weekend_avg > 0 and weekday_avg > 0:
            ratio = weekend_avg / weekday_avg
            if ratio > 1.2:
                insights.append({
                    "severity": "info",
                    "text": f"Weekend spending is {ratio:.1f}x higher than weekdays "
                            f"(\u20ac{weekend_avg:.0f} vs \u20ac{weekday_avg:.0f} avg/day).",
                })

    # --- Subscription detection ---
    sub_txns = exp[exp["category"] == "Subscriptions"]
    if len(sub_txns) > 0:
        sub_total = sub_txns["abs_amount"].sum()
        insights.append({
            "severity": "info",
            "text": f"Recurring subscriptions total: \u20ac{sub_total:,.2f} "
                    f"({len(sub_txns)} transactions).",
        })

    # --- Annual-specific: trend analysis ---
    if is_annual and len(exp) > 0:
        monthly = compute_monthly_trend(df)
        expenses_by_month = monthly["expenses"].values
        months_arr = np.arange(1, 13)
        valid = expenses_by_month > 0
        if valid.sum() >= 3:
            coeffs = np.polyfit(months_arr[valid], expenses_by_month[valid], 1)
            slope = coeffs[0]
            if abs(slope) > 10:
                direction = "increasing" if slope > 0 else "decreasing"
                insights.append({
                    "severity": "warning" if slope > 0 else "info",
                    "text": f"Expense trend is {direction} at ~\u20ac{abs(slope):.0f}/month "
                            f"over the year.",
                })

            # H1 vs H2
            h1 = expenses_by_month[:6].sum()
            h2 = expenses_by_month[6:].sum()
            if h1 > 0 and h2 > 0:
                change = (h2 - h1) / h1 * 100
                if abs(change) > 10:
                    direction = "higher" if change > 0 else "lower"
                    insights.append({
                        "severity": "info",
                        "text": f"H2 spending was {abs(change):.0f}% {direction} than H1 "
                                f"(\u20ac{h2:,.0f} vs \u20ac{h1:,.0f}).",
                    })

            # Best/worst months
            best_idx = np.argmin(expenses_by_month[valid])
            worst_idx = np.argmax(expenses_by_month[valid])
            valid_months = months_arr[valid]
            insights.append({
                "severity": "info",
                "text": f"Lowest spending month: {calendar.month_name[valid_months[best_idx]]} "
                        f"(\u20ac{expenses_by_month[valid][best_idx]:,.0f}). "
                        f"Highest: {calendar.month_name[valid_months[worst_idx]]} "
                        f"(\u20ac{expenses_by_month[valid][worst_idx]:,.0f}).",
            })

    # Sort: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 3))
    return insights


# ══════════════════════════════════════════════════════════════
# 4. VISUALIZATION LAYER
# ══════════════════════════════════════════════════════════════

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_category_pie(cat_breakdown):
    fig, ax = plt.subplots(figsize=(6, 5))
    top = cat_breakdown.head(8)
    rest = cat_breakdown.iloc[8:]
    labels = list(top["category"])
    sizes = list(top["total"])
    if len(rest) > 0:
        labels.append("Other")
        sizes.append(rest["total"].sum())
    colors = CHART_PALETTE[:len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        textprops={"fontsize": 9}, pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Spending by Category", fontsize=13, fontweight="bold", color=DARK_TEXT, pad=15)
    return _fig_to_base64(fig)


def chart_category_bar(cat_breakdown):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = cat_breakdown.head(10)
    y = range(len(top))
    ax.barh(y, top["total"], color=STEEL_BLUE, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(top["category"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Amount (\u20ac)", fontsize=10)
    ax.set_title("Top Categories by Spend", fontsize=13, fontweight="bold", color=DARK_TEXT)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"\u20ac{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_monthly_income_expenses(monthly_trend):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    months = [calendar.month_abbr[m] for m in monthly_trend["month"]]
    x = range(len(months))
    ax.bar([i - 0.2 for i in x], monthly_trend["income"], 0.4,
           label="Income", color=GREEN, edgecolor="white")
    ax.bar([i + 0.2 for i in x], monthly_trend["expenses"], 0.4,
           label="Expenses", color=RED, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=9)
    ax.set_ylabel("Amount (\u20ac)", fontsize=10)
    ax.set_title("Monthly Income vs Expenses", fontsize=13, fontweight="bold", color=DARK_TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"\u20ac{y:,.0f}"))
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_daily_spending(daily_trend):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(daily_trend["date"], daily_trend["total"], color=STEEL_BLUE, alpha=0.5, width=0.8, label="Daily")
    ax.plot(daily_trend["date"], daily_trend["ma7"], color=RED, linewidth=2, label="7-day MA")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Amount (\u20ac)", fontsize=10)
    ax.set_title("Daily Spending", fontsize=13, fontweight="bold", color=DARK_TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"\u20ac{y:,.0f}"))
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_payment_donut(pm_breakdown):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = CHART_PALETTE[:len(pm_breakdown)]
    wedges, texts, autotexts = ax.pie(
        pm_breakdown["total"], labels=pm_breakdown["method"],
        autopct="%1.1f%%", colors=colors,
        textprops={"fontsize": 9}, pctdistance=0.8,
        wedgeprops={"width": 0.4},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Payment Methods", fontsize=13, fontweight="bold", color=DARK_TEXT, pad=15)
    return _fig_to_base64(fig)


def chart_weekday_pattern(weekday_data):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [AMBER if wd >= 5 else STEEL_BLUE for wd in weekday_data["weekday"]]
    ax.bar(weekday_data["day_name"], weekday_data["avg_spend"], color=colors, edgecolor="white")
    ax.set_ylabel("Avg Spend (\u20ac)", fontsize=10)
    ax.set_title("Spending by Day of Week", fontsize=13, fontweight="bold", color=DARK_TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"\u20ac{y:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_top_merchants(merchant_breakdown):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = merchant_breakdown.head(10)
    y = range(len(top))
    ax.barh(y, top["total"], color=AMBER, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(top["merchant"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Amount (\u20ac)", fontsize=10)
    ax.set_title("Top 10 Merchants", fontsize=13, fontweight="bold", color=DARK_TEXT)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"\u20ac{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ── Savings-advice charts ─────────────────────────────────────

def _chart_savings_potential(advice_items):
    """Vertical bar: current spend vs potential savings per category."""
    n = len(advice_items)
    fig, ax = plt.subplots(figsize=(max(n * 0.65, 5), 5))
    names = [a["name"] for a in advice_items]
    totals = [a["annual"] for a in advice_items]
    potentials = [a["potential"] for a in advice_items]
    kept = [t - p for t, p in zip(totals, potentials)]

    x = np.arange(n)
    width = 0.55
    ax.bar(x, kept, width, color=STEEL_BLUE, edgecolor="white", label="Keep")
    ax.bar(x, potentials, width, bottom=kept, color=RED,
           edgecolor="white", alpha=0.7, label="Can save")

    for i, (k, p) in enumerate(zip(kept, potentials)):
        ax.text(i, k + p + 30, f"-\u20ac{p:,.0f}",
                ha="center", va="bottom", fontsize=7, color=RED)

    wrapped = [n.replace(" & ", "\n& ").replace(" ", "\n") if " " in n else n
               for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped, fontsize=8, ha="center")
    ax.set_xlim(-0.5, n - 0.5)

    ax.set_ylabel("Annual Amount (\u20ac)", fontsize=10)
    ax.set_title("Spending vs Savings Potential", fontsize=14, fontweight="bold", color=DARK_TEXT)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"\u20ac{y:,.0f}"))
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.2)
    return _fig_to_base64(fig)


def _chart_expense_breakdown(expenses, advice_items):
    """Two pie charts: current vs proposed after advice."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    names = [n for n, _ in expenses]
    values = [v for _, v in expenses]
    colors = CHART_PALETTE[:len(names)]

    savings_map = {a["name"]: a["potential"] for a in advice_items}
    proposed = [v - savings_map.get(n, 0) for n, v in expenses]

    def make_autopct(vals):
        def autopct(pct):
            amt = pct / 100.0 * sum(vals)
            return f"{pct:.1f}%\n\u20ac{amt:,.0f}"
        return autopct

    # Current
    wedges1, texts1, auto1 = ax1.pie(
        values, labels=names, autopct=make_autopct(values), colors=colors,
        textprops={"fontsize": 9}, pctdistance=0.72,
    )
    for t in auto1:
        t.set_fontsize(7)
    total_cur = sum(values)
    ax1.set_title(f"Current Expenses \u2014 \u20ac{total_cur:,.0f}",
                  fontsize=13, fontweight="bold", color=DARK_TEXT, pad=15)

    # Proposed
    wedges2, texts2, auto2 = ax2.pie(
        proposed, labels=names, autopct=make_autopct(proposed), colors=colors,
        textprops={"fontsize": 9}, pctdistance=0.72,
    )
    for t in auto2:
        t.set_fontsize(7)
    total_prop = sum(proposed)
    ax2.set_title(f"Proposed Expenses \u2014 \u20ac{total_prop:,.0f}  (save \u20ac{total_cur - total_prop:,.0f})",
                  fontsize=13, fontweight="bold", color=GREEN, pad=15)

    fig.tight_layout(pad=2.5)
    return _fig_to_base64(fig)


def _fmt_k(v):
    if v >= 1000:
        k = v / 1000
        return f"{k:.1f}K" if k != int(k) else f"{int(k)}K"
    return f"{v:.0f}"


def _chart_monthly_category(all_data, year, top_categories):
    """Horizontal stacked bar: months on Y-axis, EUR on X-axis."""
    fig, ax = plt.subplots(figsize=(6, 6))
    months = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in months]
    left = np.zeros(12)

    segments = []
    for i, (cat_name, _) in enumerate(top_categories[:6]):
        cat_data = all_data[(all_data["type"] == "Expense") &
                            (all_data["category"] == cat_name) &
                            (all_data["datetime"].dt.year == year)]
        monthly = cat_data.groupby(cat_data["datetime"].dt.month)["abs_amount"].sum()
        values = np.array([monthly.get(m, 0) for m in months])
        ax.barh(month_labels, values, left=left,
                color=CHART_PALETTE[i % len(CHART_PALETTE)],
                edgecolor="white", label=cat_name, height=0.65)
        segments.append((left.copy(), values))
        left += values

    for seg_left, seg_vals in segments:
        for j, (l, v) in enumerate(zip(seg_left, seg_vals)):
            if v > left.max() * 0.06:
                ax.text(l + v / 2, j, _fmt_k(v), ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")

    ax.set_xlim(0, left.max() * 1.05)
    ax.set_xlabel("Amount (\u20ac)", fontsize=10)
    ax.set_title("Monthly Discretionary Spending by Category",
                 fontsize=14, fontweight="bold", color=DARK_TEXT)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"\u20ac{x:,.0f}"))
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.5)
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════
# 5. SAVINGS ADVICE ENGINE
# ══════════════════════════════════════════════════════════════

def build_savings_advice(all_data, year):
    """Compute savings advice data from the DataFrame."""
    df = fetch_transactions(all_data, year)
    income = df.loc[df["type"] == "Income", "abs_amount"].sum()
    exp = df[df["type"] == "Expense"]
    expenses = (exp.groupby("category")["abs_amount"].sum()
                .sort_values(ascending=False)
                .reset_index())
    expenses.columns = ["name", "total"]
    expenses_list = list(zip(expenses["name"], expenses["total"]))
    total_exp = sum(t for _, t in expenses_list)
    surplus = income - total_exp
    savings_rate = surplus / income * 100 if income > 0 else 0

    discretionary = [(name, total) for name, total in expenses_list
                     if name not in ESSENTIAL_CATEGORIES]
    disc_total = sum(t for _, t in discretionary)

    advice_items = []
    for name, total in discretionary:
        monthly = total / 12
        pct_of_disc = total / disc_total * 100

        if pct_of_disc > 15:
            save_pct, severity, tip = 0.30, "high", "High share \u2014 aim to cut 30%"
        elif pct_of_disc > 10:
            save_pct, severity, tip = 0.20, "medium", "Moderate \u2014 try reducing 20%"
        elif monthly > 150:
            save_pct, severity, tip = 0.15, "medium", "Above EUR 150/mo \u2014 trim 15%"
        else:
            save_pct, severity, tip = 0.10, "low", "Reasonable \u2014 small cuts possible"

        potential = total * save_pct
        advice_items.append({
            "name": name, "annual": total, "monthly": monthly,
            "potential": potential, "severity": severity, "tip": tip,
            "save_pct": int(save_pct * 100),
        })

    advice_items.sort(key=lambda x: -x["potential"])
    total_potential = sum(a["potential"] for a in advice_items)
    projected_rate = (surplus + total_potential) / income * 100 if income > 0 else 0

    # Actionable tips
    cat_map = {name: total for name, total in discretionary}
    tips = []
    if "Subscriptions" in cat_map:
        tips.append(f"Review subscriptions (EUR {cat_map['Subscriptions']:,.0f}/yr) \u2014 "
                    "cancel unused services, share family plans")
    if "Food & Drink" in cat_map:
        tips.append(f"Food & Drink at EUR {cat_map['Food & Drink']/12:,.0f}/mo \u2014 "
                    "cook more at home, limit dining out to weekends")
    if "Online Shopping" in cat_map:
        tips.append(f"Online Shopping EUR {cat_map['Online Shopping']/12:,.0f}/mo \u2014 "
                    "use a 48-hour rule before purchases, unsubscribe from promo emails")
    if "Travel" in cat_map:
        tips.append(f"Travel EUR {cat_map['Travel']/12:,.0f}/mo \u2014 "
                    "book earlier, use reward points, consider off-peak travel")
    if "Car Service" in cat_map and "Car Fuel" in cat_map:
        car_total = cat_map["Car Service"] + cat_map["Car Fuel"]
        tips.append(f"Car costs EUR {car_total/12:,.0f}/mo combined \u2014 "
                    "compare service quotes, consider fuel-efficient driving habits")
    if "Groceries" in cat_map:
        tips.append(f"Groceries EUR {cat_map['Groceries']/12:,.0f}/mo \u2014 "
                    "plan meals weekly, buy store brands, use discount apps")

    return {
        "income": income, "total_exp": total_exp, "surplus": surplus,
        "savings_rate": savings_rate, "disc_total": disc_total,
        "advice_items": advice_items, "total_potential": total_potential,
        "projected_rate": projected_rate, "tips": tips,
        "expenses": expenses_list, "discretionary": discretionary,
    }


# ══════════════════════════════════════════════════════════════
# 6. HTML TEMPLATES
# ══════════════════════════════════════════════════════════════

HTML_TEMPLATE = dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  @page { size: A4; margin: 1.5cm; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: {{ dark_text }};
    background: {{ light_bg }};
    line-height: 1.5;
    font-size: 14px;
    padding: 20px;
  }
  .container { max-width: 1000px; margin: 0 auto; background: {{ white }}; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }

  /* Header */
  .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 3px solid {{ steel_blue }}; }
  .header h1 { color: {{ steel_blue }}; font-size: 28px; margin-bottom: 5px; }
  .header .subtitle { color: #666; font-size: 15px; }
  .header .generated { color: #999; font-size: 12px; margin-top: 5px; }

  /* Summary cards */
  .summary-cards { display: flex; gap: 15px; margin-bottom: 30px; flex-wrap: wrap; }
  .card { flex: 1; min-width: 200px; padding: 20px; border-radius: 8px; text-align: center; }
  .card .label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85; margin-bottom: 5px; }
  .card .value { font-size: 26px; font-weight: 700; }
  .card-income { background: #e8f5e9; color: {{ green }}; }
  .card-expenses { background: #fce4ec; color: {{ red }}; }
  .card-net { background: #e3f2fd; color: {{ steel_blue }}; }
  .card-count { background: #fff8e1; color: #f57f17; }

  /* Section */
  .section { margin-bottom: 30px; page-break-inside: avoid; }
  .section h2 { color: {{ steel_blue }}; font-size: 20px; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #eee; }

  /* Insights */
  .insight-box { padding: 12px 15px; margin-bottom: 10px; border-left: 4px solid; border-radius: 4px; font-size: 13px; }
  .insight-info { border-color: {{ green }}; background: #f0faf3; }
  .insight-warning { border-color: {{ amber }}; background: #fff9e6; }
  .insight-critical { border-color: {{ red }}; background: #fef0ef; }
  .insight-severity { font-weight: 700; text-transform: uppercase; font-size: 11px; margin-right: 8px; }
  .insight-severity-info { color: {{ green }}; }
  .insight-severity-warning { color: {{ amber }}; }
  .insight-severity-critical { color: {{ red }}; }

  /* Charts */
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .chart-full { grid-column: 1 / -1; }
  .chart-box { text-align: center; }
  .chart-box img { max-width: 100%; height: auto; border-radius: 6px; }

  /* Tables */
  table { width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 13px; }
  thead th { background: {{ steel_blue }}; color: white; padding: 10px 12px; text-align: left; font-weight: 600; }
  tbody tr:nth-child(even) { background: #f5f7fa; }
  tbody td { padding: 8px 12px; border-bottom: 1px solid #eee; }
  .text-right { text-align: right; }

  /* Print */
  @media print {
    body { background: white; padding: 0; }
    .container { box-shadow: none; padding: 0; }
    .section { page-break-inside: avoid; }
    .charts-grid { page-break-inside: avoid; }
  }
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>{{ title }}</h1>
    <div class="subtitle">{{ subtitle }}</div>
    <div class="generated">Generated on {{ generated_date }}</div>
  </div>

  <!-- Summary Cards -->
  <div class="summary-cards">
    <div class="card card-income">
      <div class="label">Total Income</div>
      <div class="value">&euro;{{ "{:,.2f}".format(summary.income) }}</div>
    </div>
    <div class="card card-expenses">
      <div class="label">Total Expenses</div>
      <div class="value">&euro;{{ "{:,.2f}".format(summary.expenses) }}</div>
    </div>
    <div class="card card-net">
      <div class="label">Net Savings</div>
      <div class="value">&euro;{{ "{:,.2f}".format(summary.net) }}</div>
    </div>
    <div class="card card-count">
      <div class="label">Transactions</div>
      <div class="value">{{ summary.count }}</div>
    </div>
  </div>

  <!-- Insights -->
  {% if insights %}
  <div class="section">
    <h2>Automated Insights</h2>
    {% for ins in insights %}
    <div class="insight-box insight-{{ ins.severity }}">
      <span class="insight-severity insight-severity-{{ ins.severity }}">{{ ins.severity }}</span>
      {{ ins.text }}
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- Charts -->
  <div class="section">
    <h2>Visual Analysis</h2>
    <div class="charts-grid">
      {% if chart_monthly_ie %}
      <div class="chart-box chart-full">
        <img src="data:image/png;base64,{{ chart_monthly_ie }}" alt="Monthly Income vs Expenses">
      </div>
      {% endif %}
      {% if chart_daily %}
      <div class="chart-box chart-full">
        <img src="data:image/png;base64,{{ chart_daily }}" alt="Daily Spending">
      </div>
      {% endif %}
      <div class="chart-box">
        <img src="data:image/png;base64,{{ chart_cat_pie }}" alt="Category Pie">
      </div>
      <div class="chart-box">
        <img src="data:image/png;base64,{{ chart_payment }}" alt="Payment Methods">
      </div>
      <div class="chart-box chart-full">
        <img src="data:image/png;base64,{{ chart_cat_bar }}" alt="Category Bar">
      </div>
      <div class="chart-box">
        <img src="data:image/png;base64,{{ chart_weekday }}" alt="Weekday Pattern">
      </div>
      <div class="chart-box">
        <img src="data:image/png;base64,{{ chart_merchants }}" alt="Top Merchants">
      </div>
    </div>
  </div>

  <!-- Category Breakdown Table -->
  <div class="section">
    <h2>Category Breakdown</h2>
    <table>
      <thead>
        <tr><th>Category</th><th class="text-right">Amount</th><th class="text-right">%</th><th class="text-right">Transactions</th></tr>
      </thead>
      <tbody>
        {% for _, row in cat_breakdown.iterrows() %}
        <tr>
          <td>{{ row.category }}</td>
          <td class="text-right">&euro;{{ "{:,.2f}".format(row.total) }}</td>
          <td class="text-right">{{ row.pct }}%</td>
          <td class="text-right">{{ row["count"] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Top Merchants Table -->
  <div class="section">
    <h2>Top Merchants</h2>
    <table>
      <thead>
        <tr><th>Merchant</th><th class="text-right">Amount</th><th class="text-right">%</th><th class="text-right">Transactions</th></tr>
      </thead>
      <tbody>
        {% for _, row in merchant_breakdown.head(15).iterrows() %}
        <tr>
          <td>{{ row.merchant }}</td>
          <td class="text-right">&euro;{{ "{:,.2f}".format(row.total) }}</td>
          <td class="text-right">{{ row.pct }}%</td>
          <td class="text-right">{{ row["count"] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Payment Methods Table -->
  <div class="section">
    <h2>Payment Methods</h2>
    <table>
      <thead>
        <tr><th>Method</th><th class="text-right">Amount</th><th class="text-right">%</th><th class="text-right">Transactions</th></tr>
      </thead>
      <tbody>
        {% for _, row in pm_breakdown.iterrows() %}
        <tr>
          <td>{{ row.method }}</td>
          <td class="text-right">&euro;{{ "{:,.2f}".format(row.total) }}</td>
          <td class="text-right">{{ row.pct }}%</td>
          <td class="text-right">{{ row["count"] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Avg Daily Spend -->
  <div class="section">
    <h2>Summary Statistics</h2>
    <table>
      <thead><tr><th>Metric</th><th class="text-right">Value</th></tr></thead>
      <tbody>
        <tr><td>Average Daily Spend</td><td class="text-right">&euro;{{ "{:,.2f}".format(summary.avg_daily_spend) }}</td></tr>
        <tr><td>Savings Rate</td><td class="text-right">{{ "{:.1f}".format(summary.net / summary.income * 100 if summary.income > 0 else 0) }}%</td></tr>
        <tr><td>Expense Transactions</td><td class="text-right">{{ expense_count }}</td></tr>
        <tr><td>Income Transactions</td><td class="text-right">{{ income_count }}</td></tr>
      </tbody>
    </table>
  </div>

</div>
</body>
</html>
""")


ADVICE_HTML_TEMPLATE = dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Savings Advice Report</title>
<style>
  @page { size: A4; margin: 1.5cm; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #2C3E50; background: #F8F9FA; line-height: 1.6; font-size: 14px; padding: 20px;
  }
  .container { max-width: 1000px; margin: 0 auto; background: #fff; padding: 35px;
    border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }

  .header { text-align: center; margin-bottom: 35px; padding-bottom: 22px;
    border-bottom: 3px solid #2E86AB; }
  .header h1 { color: #2E86AB; font-size: 28px; margin-bottom: 6px; }
  .header .subtitle { color: #666; font-size: 15px; }
  .header .generated { color: #999; font-size: 12px; margin-top: 4px; }

  .cards { display: flex; gap: 15px; margin-bottom: 30px; flex-wrap: wrap; }
  .card { flex: 1; min-width: 140px; padding: 18px 14px; border-radius: 10px;
    text-align: center; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    opacity: 0.85; margin-bottom: 4px; }
  .card .value { font-size: 24px; font-weight: 700; }
  .card-income { background: #e8f5e9; color: #27AE60; }
  .card-expenses { background: #fce4ec; color: #E74C3C; }
  .card-surplus { background: #e3f2fd; color: #2E86AB; }
  .card-rate { background: #fff8e1; color: #F57F17; }

  .section { margin-bottom: 32px; page-break-inside: avoid; }
  .section h2 { color: #2E86AB; font-size: 20px; margin-bottom: 15px;
    padding-bottom: 8px; border-bottom: 2px solid #eee; }

  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .chart-full { grid-column: 1 / -1; }
  .chart-box { text-align: center; overflow: hidden; }
  .chart-box img { display: block; max-width: 50%; height: auto; border-radius: 8px; margin: 0 auto; }
  .chart-wide img { max-width: 80%; }
  .chart-centered { max-width: 600px; margin: 0 auto; }
  .chart-centered img { display: block; max-width: 100%; height: auto; border-radius: 8px; margin: 0 auto; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 15px; }
  thead th { background: #2E86AB; color: #fff; padding: 10px 12px; text-align: left; font-weight: 600; }
  tbody tr:nth-child(even) { background: #f5f7fa; }
  tbody td { padding: 9px 12px; border-bottom: 1px solid #eee; }
  .text-right { text-align: right; }

  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; }
  .badge-high { background: #fce4ec; color: #E74C3C; }
  .badge-medium { background: #fff8e1; color: #E67E22; }
  .badge-low { background: #e8f5e9; color: #27AE60; }

  .tip-card { display: flex; align-items: flex-start; gap: 14px; padding: 14px 18px;
    margin-bottom: 12px; background: #f0f7ff; border-left: 4px solid #2E86AB;
    border-radius: 0 8px 8px 0; font-size: 13.5px; }
  .tip-num { background: #2E86AB; color: #fff; width: 28px; height: 28px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13px; flex-shrink: 0; }

  .projection { background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 100%);
    border-radius: 10px; padding: 22px 28px; margin-top: 20px; }
  .projection h3 { color: #2E86AB; margin-bottom: 10px; font-size: 16px; }
  .projection .row { display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px dashed #ccc; }
  .projection .row:last-child { border: none; font-weight: 700; font-size: 16px; }
  .projection .val-green { color: #27AE60; }
  .projection .val-amber { color: #F57F17; }

  @media print {
    body { background: #fff; padding: 0; }
    .container { box-shadow: none; padding: 0; }
  }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Savings Advice Report</h1>
    <div class="subtitle">Personalized insights to optimize your finances</div>
    <div class="generated">Generated on {{ generated_date }}</div>
  </div>

  <div class="cards">
    <div class="card card-income">
      <div class="label">Annual Income</div>
      <div class="value">&euro;{{ "{:,.0f}".format(income) }}</div>
    </div>
    <div class="card card-expenses">
      <div class="label">Annual Expenses</div>
      <div class="value">&euro;{{ "{:,.0f}".format(total_exp) }}</div>
    </div>
    <div class="card card-surplus">
      <div class="label">Current Surplus</div>
      <div class="value">&euro;{{ "{:,.0f}".format(surplus) }}</div>
    </div>
    <div class="card card-rate">
      <div class="label">Savings Rate</div>
      <div class="value">{{ "{:.1f}".format(savings_rate) }}%</div>
    </div>
  </div>

  <div class="section">
    <h2>Visual Analysis</h2>
    <div class="charts-grid">
      <div class="chart-box chart-full">
        <img src="data:image/png;base64,{{ chart_potential }}" alt="Savings Potential">
      </div>
      <div class="chart-box chart-full chart-wide">
        <img src="data:image/png;base64,{{ chart_pie }}" alt="Expense Breakdown">
      </div>
      <div class="chart-box chart-full">
        <img src="data:image/png;base64,{{ chart_monthly }}" alt="Monthly by Category">
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Category-by-Category Advice</h2>
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th class="text-right">Annual</th>
          <th class="text-right">Monthly</th>
          <th class="text-right">Cut&nbsp;%</th>
          <th class="text-right">Potential Saving</th>
          <th>Priority</th>
          <th>Advice</th>
        </tr>
      </thead>
      <tbody>
        {% for a in advice_items %}
        <tr>
          <td><strong>{{ a.name }}</strong></td>
          <td class="text-right">&euro;{{ "{:,.2f}".format(a.annual) }}</td>
          <td class="text-right">&euro;{{ "{:,.2f}".format(a.monthly) }}</td>
          <td class="text-right">{{ a.save_pct }}%</td>
          <td class="text-right"><strong>&euro;{{ "{:,.2f}".format(a.potential) }}</strong></td>
          <td><span class="badge badge-{{ a.severity }}">{{ a.severity }}</span></td>
          <td>{{ a.tip }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="projection">
      <h3>If You Follow All Advice</h3>
      <div class="row">
        <span>Current annual surplus</span>
        <span class="val-amber">&euro;{{ "{:,.2f}".format(surplus) }}</span>
      </div>
      <div class="row">
        <span>Total potential savings</span>
        <span class="val-green">+ &euro;{{ "{:,.2f}".format(total_potential) }}</span>
      </div>
      <div class="row">
        <span>Projected annual surplus</span>
        <span class="val-green">&euro;{{ "{:,.2f}".format(surplus + total_potential) }}</span>
      </div>
      <div class="row">
        <span>Projected savings rate</span>
        <span class="val-green">{{ "{:.1f}".format(projected_rate) }}%</span>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Actionable Tips</h2>
    {% for tip in tips %}
    <div class="tip-card">
      <div class="tip-num">{{ loop.index }}</div>
      <div>{{ tip }}</div>
    </div>
    {% endfor %}
  </div>

</div>
</body>
</html>
""")


# ══════════════════════════════════════════════════════════════
# 7. RENDERING
# ══════════════════════════════════════════════════════════════

def render_html(template_str, data):
    template = Template(template_str)
    return template.render(**data)


def _patch_cffi_for_homebrew():
    """Monkey-patch cffi.FFI.dlopen to find Homebrew libs on macOS."""
    lib_dir = "/opt/homebrew/lib"
    if not os.path.isdir(lib_dir):
        return
    try:
        import cffi
    except ImportError:
        return
    original_dlopen = cffi.FFI.dlopen

    def patched_dlopen(self, name, flags=0):
        try:
            return original_dlopen(self, name, flags)
        except OSError:
            for suffix in ("", ".dylib"):
                full = os.path.join(lib_dir, name + suffix)
                if os.path.exists(full):
                    return original_dlopen(self, full, flags)
            parts = name.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                dotname = parts[0] + "." + parts[1] + ".dylib"
                full = os.path.join(lib_dir, dotname)
                if os.path.exists(full):
                    return original_dlopen(self, full, flags)
            raise

    cffi.FFI.dlopen = patched_dlopen

_patch_cffi_for_homebrew()


def render_pdf(html_content, output_path):
    try:
        from weasyprint import HTML as WeasyHTML
    except (ImportError, OSError) as e:
        print(f"  WeasyPrint unavailable ({e}) — skipping PDF generation.")
        return False
    WeasyHTML(string=html_content).write_pdf(output_path)
    return True


# ══════════════════════════════════════════════════════════════
# 8. REPORT BUILDERS
# ══════════════════════════════════════════════════════════════

def build_finance_report(all_data, year, month=None, output_dir=None, fmt="both"):
    """Build annual or monthly finance report."""
    output_dir = output_dir or DEFAULT_REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    is_annual = month is None

    print(f"Loading transactions for {year}" + (f"-{month:02d}" if month else "") + "...")
    df = fetch_transactions(all_data, year, month)
    if len(df) == 0:
        print("No transactions found for this period.")
        return

    print(f"  Found {len(df)} transactions.")
    prev_df = fetch_previous_period(all_data, year, month)
    print(f"  Previous period: {len(prev_df)} transactions.")

    # Analysis
    print("Computing analysis...")
    summary_dict = compute_summary(df)
    summary = type("S", (), summary_dict)()
    cat_breakdown = compute_category_breakdown(df)
    merchant_breakdown = compute_merchant_breakdown(df)
    pm_breakdown = compute_payment_method_breakdown(df)
    daily_trend = compute_daily_trend(df)
    monthly_trend = compute_monthly_trend(df) if is_annual else None
    weekday_data = compute_weekday_pattern(df)

    # Insights
    print("Generating insights...")
    insights = generate_insights(df, prev_df, summary_dict, cat_breakdown, merchant_breakdown, is_annual)

    # Charts
    print("Rendering charts...")
    charts = {
        "chart_cat_pie": chart_category_pie(cat_breakdown),
        "chart_cat_bar": chart_category_bar(cat_breakdown),
        "chart_payment": chart_payment_donut(pm_breakdown),
        "chart_weekday": chart_weekday_pattern(weekday_data),
        "chart_merchants": chart_top_merchants(merchant_breakdown),
        "chart_monthly_ie": chart_monthly_income_expenses(monthly_trend) if is_annual else None,
        "chart_daily": chart_daily_spending(daily_trend) if not is_annual else None,
    }

    # Build title
    if is_annual:
        title = f"Annual Finance Report {year}"
        subtitle = f"Full year overview \u2014 January to December {year}"
        file_stem = f"finance_report_{year}_annual"
    else:
        month_name = calendar.month_name[month]
        title = f"Monthly Finance Report \u2014 {month_name} {year}"
        subtitle = f"{month_name} {year} detailed breakdown"
        file_stem = f"finance_report_{year}_{month:02d}_{month_name.lower()}"

    expense_count = len(df[df["type"] == "Expense"])
    income_count = len(df[df["type"] == "Income"])

    template_data = {
        "title": title,
        "subtitle": subtitle,
        "generated_date": datetime.now().strftime("%B %d, %Y at %H:%M"),
        "summary": summary,
        "insights": insights,
        "cat_breakdown": cat_breakdown,
        "merchant_breakdown": merchant_breakdown,
        "pm_breakdown": pm_breakdown,
        "expense_count": expense_count,
        "income_count": income_count,
        "steel_blue": STEEL_BLUE, "amber": AMBER, "dark_text": DARK_TEXT,
        "green": GREEN, "red": RED, "light_bg": LIGHT_BG, "white": WHITE,
        **charts,
    }

    # Render HTML
    print("Building HTML...")
    html_content = render_html(HTML_TEMPLATE, template_data)

    if fmt in ("html", "both"):
        html_path = os.path.join(output_dir, file_stem + ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"  HTML saved: {html_path}")

    if fmt in ("pdf", "both"):
        pdf_path = os.path.join(output_dir, file_stem + ".pdf")
        print("  Generating PDF...")
        if render_pdf(html_content, pdf_path):
            print(f"  PDF saved:  {pdf_path}")

    print("Done!")


def build_savings_report(all_data, year, output_dir=None, fmt="both"):
    """Build savings advice report."""
    output_dir = output_dir or DEFAULT_REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analysing spending for {year}...")
    data = build_savings_advice(all_data, year)

    # Charts
    print("Rendering charts...")
    chart_potential = _chart_savings_potential(data["advice_items"])
    chart_pie = _chart_expense_breakdown(data["expenses"], data["advice_items"])
    chart_monthly = _chart_monthly_category(all_data, year, data["discretionary"])

    template_data = {
        "generated_date": datetime.now().strftime("%B %d, %Y at %H:%M"),
        "chart_potential": chart_potential,
        "chart_pie": chart_pie,
        "chart_monthly": chart_monthly,
        **data,
    }

    print("Building HTML...")
    html_content = render_html(ADVICE_HTML_TEMPLATE, template_data)
    file_stem = f"savings_advice_{year}"

    if fmt in ("html", "both"):
        html_path = os.path.join(output_dir, file_stem + ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"  HTML saved: {html_path}")

    if fmt in ("pdf", "both"):
        pdf_path = os.path.join(output_dir, file_stem + ".pdf")
        print("  Generating PDF...")
        if render_pdf(html_content, pdf_path):
            print(f"  PDF saved:  {pdf_path}")

    print("Done!")


# ══════════════════════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Standalone finance report generator (CSV-based, no database required)")
    parser.add_argument("--year", type=int, required=True, help="Report year (e.g. 2024)")
    parser.add_argument("--month", type=int, default=None, help="Month number (1-12); omit for annual")
    parser.add_argument("--report", choices=["annual", "monthly", "savings"], default=None,
                        help="Report type (default: annual, or monthly if --month is set)")
    parser.add_argument("--format", choices=["html", "pdf", "both"], default="both",
                        help="Output format (default: both)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV data file")
    args = parser.parse_args()

    if args.month and not (1 <= args.month <= 12):
        print("Error: --month must be 1-12")
        sys.exit(1)

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    # Determine report type
    report_type = args.report
    if report_type is None:
        report_type = "monthly" if args.month else "annual"

    # Load all data once
    print(f"Reading CSV: {args.csv}")
    all_data = load_csv(args.csv)
    print(f"  {len(all_data)} records loaded.\n")

    if report_type == "savings":
        build_savings_report(all_data, args.year, args.output_dir, args.format)
    else:
        build_finance_report(all_data, args.year, args.month, args.output_dir, args.format)


if __name__ == "__main__":
    main()

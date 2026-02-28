# Personal Finance Report Generator

A standalone Python tool that analyses personal finance transactions from a CSV file and generates rich HTML/PDF reports with charts, automated insights, and savings advice.

No database required — everything runs locally from a single CSV file.

---

## Features

- **Annual Report** — full-year overview: income vs expenses, category/merchant breakdowns, 7 chart types, automated anomaly detection, trend analysis
- **Monthly Report** — single-month deep dive with daily spending trends, period-over-period comparisons
- **Savings Advice Report** — discretionary spending analysis, per-category cut recommendations, projected savings rate, actionable tips

All reports are self-contained HTML files (charts embedded as base64 images). Optional PDF export via WeasyPrint.

---

## Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)

---

## Installation

### 1. Clone or download the project

```bash
git clone <repository-url>
cd financeAssistent_standalone
```

Or download and extract the ZIP archive, then open a terminal in the project folder.

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `matplotlib`, `numpy`, `jinja2`, and `weasyprint`.

> **Note:** WeasyPrint is only needed for PDF output. If you only need HTML reports, you can skip it:
> ```bash
> pip install pandas matplotlib numpy jinja2
> ```

---

## Usage

All commands are run from the project root directory.

### Annual report (HTML + PDF)

```bash
python3 finance_report.py --year 2024
```

### Monthly report

```bash
python3 finance_report.py --year 2024 --month 6
```

### Savings advice report

```bash
python3 finance_report.py --year 2024 --report savings
```

### HTML only (skip PDF)

```bash
python3 finance_report.py --year 2024 --format html
```

### PDF only

```bash
python3 finance_report.py --year 2024 --format pdf
```

### Custom output directory

```bash
python3 finance_report.py --year 2024 --output-dir /path/to/output
```

### Use a different CSV file

```bash
python3 finance_report.py --csv data/my_transactions.csv --year 2025
```

---

## CSV Format

The input CSV must have the following columns:

| Column          | Description                    | Example                |
|-----------------|--------------------------------|------------------------|
| DateTime        | Transaction date and time      | `1/15/24 14:30`        |
| Description     | Transaction description        | `Monthly Salary`       |
| Merchant        | Merchant or payee name         | `Tech Solutions Ltd`   |
| Category_Code   | Short category code            | `SAL`                  |
| Category_Name   | Full category name             | `Salary`               |
| Type            | `Income` or `Expense`          | `Expense`              |
| Amount_EUR      | Amount in EUR (negative = expense) | `-45.99`           |
| Payment_Method  | Payment method used            | `Credit Card`          |

A sample dataset with 1,500 transactions is included in `data/`.

---

## Project Structure

```
financeAssistent_standalone/
├── finance_report.py       # Main script — all report logic in one file
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   └── personal_finance_2024_1500_transactions.csv
└── reports/                # Generated reports appear here
```

---

## Troubleshooting

**WeasyPrint fails to install or run**

WeasyPrint requires system-level libraries (Cairo, Pango, GDK-PixBuf). If installation fails:

- **macOS:** `brew install cairo pango gdk-pixbuf libffi`
- **Ubuntu/Debian:** `sudo apt install libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev`
- **Windows:** See [WeasyPrint installation docs](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)

Or skip PDF and use HTML-only mode: `--format html`

**No transactions found for this period**

Make sure the `--year` (and `--month`) match dates present in your CSV file. The sample data covers January–December 2024.

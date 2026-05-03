# Predictive Liquidity & AR Manager

A financial-grade Power BI application powered by Python, SQL, and R to eliminate cashflow blind spots. While standard financial reports only show what has happened, this dashboard predicts what will happen. It exposes exactly where cash is stuck in the Accounts Receivable (AR) pipeline, calculates customer risk scores, and forecasts liquidity for the next 90 days.

## The Backstory: Moving Beyond Excel

In corporate finance, the CFO's biggest nightmare is a lack of cash visibility. Standard Excel exports and static dashboards fail to answer the most critical question: "When exactly will the money hit the bank?" I built this project to bridge the gap between traditional financial reporting and advanced data science. By combining robust SQL data structures, Python-based forecasting, and statistical R-visuals, this tool transitions financial analysis from reactive reporting to proactive risk management.

## Technical Architecture

This project implements a full-cycle, predictive data pipeline:

*   **Data Source:** Olist Brazilian E-Commerce Public Dataset (via Kaggle), utilizing orders, payments, and customer tables to simulate a complex invoice and payment environment.
*   **ETL and Data Engineering (SQL):** Cleaned and joined multiple relational tables into a flat reporting view. Engineered the target variable Payment_Delay (Actual Payment Date - Planned Payment Date) to serve as the foundation for the predictive models.
*   **Machine Learning (Python):** Utilized Prophet (or ExponentialSmoothing) in a Jupyter Notebook to generate a 90-day cashflow forecast (cashflow_forecast.csv) and calculated historical Customer Risk Scores.
*   **Statistical Analysis (R / ggplot2):** Integrated custom R scripts directly into Power BI to render high-density probability plots of payment delays.
*   **Workflow Note:** Used AI-assisted "vibe coding" to seamlessly connect the SQL, Python, and R logic, building upon foundational data engineering knowledge.

---

## The Dashboards

### 1. Liquidity Radar (Strategic Overview)
*   **Executive KPIs:** Real-time tracking of Total Receivables, DSO (Days Sales Outstanding), and 90-Day Forecasted Cash.
*   **The Cashflow Chart:** A combined column chart contrasting historical actuals against the Python-generated machine learning forecast.
*   **Liquidity Gauges:** Quick-glance indicators to assess current liquidity health against corporate targets.

### 2. AR Ageing (Operational View)
*   **Receivables Buckets:** Classic AR ageing classification (0-30, 31-60, 60+ days) to identify aging debt at a glance.
*   **Workflow Prioritization:** Highlights the most critical outstanding balances to streamline the collections process.

### 3. Customer Insights (Deep-Dive Drill-Through)
*   **Behavioral Profiling:** Accessible via drill-through from the main pages. Analyzes individual customer payment habits (e.g., identifying if a customer consistently pays on specific days).
*   **Account History:** A detailed log of individual invoices, contact history, and total outstanding obligations.

### 4. Risk Analysis
*   **Probability Density (R Visual):** Features a custom ggplot2 chart visualizing the statistical probability of a customer paying 5, 10, or 20 days late.
*   **What-If Simulations:** Interactive parameters allowing the CFO to stress-test liquidity. (e.g., "What happens to our cash runway if all outstanding payments are delayed by an additional 10 days?").

### 5. Technical Docs
*   **Code Transparency:** Cleanly formatted insights and snippets into the underlying SQL data transformations and the Python forecasting scripts.

---

## Key Insights

*   **The Delay Pattern:** Statistical analysis via R reveals that minor delays are highly predictable based on customer region and historical behavior, allowing for targeted early-intervention.
*   **Proactive Cashflow:** By overlaying the Python forecast with actuals, the business can anticipate liquidity dips up to 3 months in advance, completely changing how short-term investments or credit lines are managed.

---

## How to Use

1.  Clone this repository.
2.  Initialize the SQL database using the provided data processing scripts.
3.  Run the Python Jupyter Notebook to generate the `cashflow_forecast.csv`.
4.  Open the Power BI file (Ensure your local R provider is enabled to render the custom ggplot2 visuals).

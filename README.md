# AWS EC2 Pricing Landscape: A Multidimensional View of the Cloud Compute Catalog

**Final project — Data Visualization, Spring 2026 — Jawaad Jariwala**

## Project Abstract

This project treats the Amazon Web Services (AWS) EC2 price list as a multidimensional dataset and asks what the *shape* of that dataset reveals about how AWS prices its compute capacity. Using the canonical AWS Price List Bulk API (accessed through the publicly mirrored aggregation maintained by Vantage at `https://instances.vantage.sh/instances.json`), the analysis ingests pricing for every EC2 instance type in every commercial region on every supported operating system (318,604 price points across 1,337 instance types and 104 regions) into a tidy long-format DataFrame. Six techniques from class are then applied: pandas DataFrame masking and joining for tidying, NumPy vectorized Pareto-dominance testing for cost-efficiency frontier extraction, principal component analysis on the standardized spec matrix to project the catalog into two dimensions, polynomial regression on log-transformed cost across instance generations, geographic visualization of a per-region price index, and an interactive Plotly explorer for individual instance lookup. The actionable insights are (i) only 6.23% of EC2 instance types are Pareto-optimal in `(vCPU, memory, price)` space, and AWS Graviton (AWS's ARM-based server CPU) populates that frontier 3.14 times as densely as Intel, (ii) AWS regional pricing has a structured geography in which Hyderabad is 25.17% cheaper than `us-east-1` and São Paulo is 28.57% more expensive, and (iii) the per-vCPU cost reduction commonly attributed to "refreshing to the latest instance generation" is in fact a product of architecture migration to Graviton (15.19% to 19.79% savings over Intel at matched generation and size), not of generational pricing alone. The deliverable is a Jupyter notebook (`notebook.ipynb`) that imports all reusable analysis code from modules under `src/` and runs unmodified in a local Python environment, in GitHub Codespaces, or in Google Colab.

## How to run

### In GitHub Codespaces (zero-touch)
1. Click the green **`<> Code`** button on this repo's GitHub page → **Codespaces** tab → **Create codespace on main**.
2. Wait ~60 seconds for the container to build. The included `.devcontainer/devcontainer.json` automatically installs all dependencies from `requirements.txt`.
3. Open `notebook.ipynb` from the file explorer and click **Run All**.

### In Google Colab
1. Upload the project folder (or `git clone` it inside Colab).
2. Open `notebook.ipynb` and select **Runtime → Run all**. The first cell verifies that `src/` is present and adds it to the Python path. Required packages (`pandas`, `numpy`, `matplotlib`, `plotly`, `scikit-learn`, `requests`, `pyarrow`) are pre-installed on Colab.

### Locally
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook
```

### Re-fetching pricing data
The repository ships with a pre-computed `data/ec2_pricing_tidy.parquet` cache (~3 MB). To rebuild from scratch from the AWS Price List mirror:
```bash
.venv/bin/python -m src.pricing_loader
```

## Repository layout
```
notebook.ipynb           # the project deliverable (executed)
requirements.txt
data/
  ec2_pricing_tidy.parquet   # cached tidy pricing DataFrame (~3 MB)
src/
  pricing_loader.py            # download + parse AWS Price List
  feature_extraction.py        # parse instance type names + processor labels
  efficiency_frontier.py       # NumPy-vectorized Pareto-dominance computation
  geo_pricing.py               # regional aggregation + region coordinates
  generation_analysis.py       # generation deflation + vendor comparison
  pca_analysis.py              # standardized PCA on instance spec matrix
  plots.py                     # matplotlib helpers
  interactive_plots.py         # Plotly map + instance explorer
```

## Data source
- AWS. *AWS Price List Bulk API.* https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html
- Vantage. *EC2 Instances Comparison.* https://instances.vantage.sh — used as the aggregated mirror so the analysis runs in seconds rather than gigabytes-per-run.

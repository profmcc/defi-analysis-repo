# Quick Start Guide

## Setup (5 minutes)

1. **Navigate to the repository**:
   ```bash
   cd defi-analysis-repo
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

5. **Open a notebook** from the `notebooks/` directory and run all cells.

## Running Notebooks

All notebooks are configured to read data from the `data/` directory using relative paths. Simply:

1. Ensure data files are in `data/` directory
2. Open notebook from `notebooks/` directory
3. Run cells sequentially

## Output Files

Most notebooks save output files. By default, some notebooks save outputs to `~/Downloads/`. You can modify the output paths in the notebook configuration sections if you prefer to save outputs to the `data/` directory or another location.

## Data Files Checklist

Before running notebooks, verify you have:

- ✅ `data/twaps.csv` (for master_twap_threshold_analysis.ipynb)
- ✅ `data/foundation_twap_sales.csv` (for master_twap_threshold_analysis.ipynb)
- ✅ `data/viewblock_thorchain_*.csv` and `*.json` files (for thorchain_data_combiner.ipynb)
- ✅ `data/swap_transactions_*.csv` files (for chainflip_volume_analyzer.ipynb and butterswap_data_analyzer.ipynb)

## Git Upload

The repository is ready for git upload:

```bash
cd defi-analysis-repo
git add .
git commit -m "Initial commit: DeFi analysis notebooks and data"
git remote add origin <your-repo-url>
git push -u origin master  # or main, depending on your default branch
```

**Note**: The `.gitignore` file is configured to exclude cache files, virtual environments, and other temporary files. Data files are included by default. If you prefer not to commit data files, uncomment the data file patterns in `.gitignore`.


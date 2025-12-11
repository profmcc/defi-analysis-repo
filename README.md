# DeFi Analysis Repository

This repository contains Jupyter notebooks for analyzing DeFi protocol data including TWAP (Time-Weighted Average Price) analysis, ButterSwap transactions, THORChain data, and Chainflip volume analysis.

## Repository Structure

```
defi-analysis-repo/
├── notebooks/              # Jupyter notebooks for analysis
│   ├── master_twap_threshold_analysis.ipynb
│   ├── butterswap_data_analyzer.ipynb
│   ├── thorchain_data_combiner.ipynb
│   └── chainflip_volume_analyzer.ipynb
├── data/                   # Data files (CSV, JSON)
│   ├── twaps.csv
│   ├── foundation_twap_sales.csv
│   ├── viewblock_thorchain_*.csv
│   ├── viewblock_thorchain_*.json
│   └── swap_transactions_*.csv
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip (Python package manager)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd defi-analysis-repo
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Start Jupyter Notebook:
```bash
jupyter notebook
```

Or use JupyterLab:
```bash
jupyter lab
```

## Notebooks Overview

### 1. Master TWAP Threshold Analysis (`master_twap_threshold_analysis.ipynb`)

**Purpose**: Comprehensive analysis of FOX price impact from both Shapeshift DAO TWAPs and Foundation TWAPs using threshold-based regression analysis.

**Key Features**:
- Loads FOX market data, DAO TWAPs, and Foundation TWAPs
- Converts Foundation TWAPs to FOX-denominated amounts
- Computes 7-day normalized volume and threshold-based splits
- Estimates multiple regression models by volume threshold
- Analyzes combined impact of DAO + Foundation TWAPs
- Includes robustness checks (Newey-West, SARIMAX, Random Forest)
- Comprehensive visualizations

**Required Data Files**:
- `data/twaps.csv` - DAO TWAPs CSV file
- `data/foundation_twap_sales.csv` - Foundation TWAPs CSV

**Dependencies**:
- CoinGecko API key (configured in notebook)
- pandas, numpy, statsmodels, matplotlib, scikit-learn, arch

**Usage**:
1. Ensure data files are in the `data/` directory
2. Open the notebook in Jupyter
3. Run all cells sequentially
4. Results will be displayed inline and saved to cache files

---

### 2. ButterSwap Data Analyzer (`butterswap_data_analyzer.ipynb`)

**Purpose**: Analyzes ButterSwap transaction data, combining multiple CSV files and performing statistical analysis.

**Key Features**:
- Combines multiple ButterSwap transaction CSV files
- Cleans and deduplicates data
- Performs statistical analysis
- Generates summary statistics

**Required Data Files**:
- ButterSwap transaction CSV files in `data/` directory
- Excel file: "Butter Affiliate Manager Dec 2 2025.xlsx" (if available)

**Dependencies**:
- pandas, numpy, pathlib, glob

**Usage**:
1. Place ButterSwap transaction CSV files in `data/` directory
2. Open the notebook in Jupyter
3. Run all cells sequentially
4. Review the combined and analyzed data

---

### 3. THORChain Data Combiner (`thorchain_data_combiner.ipynb`)

**Purpose**: Combines multiple THORChain data files (CSV and JSON) from ViewBlock, cleans the data, removes duplicates, and prepares it for analysis.

**Key Features**:
- Loads multiple ViewBlock THORChain data files
- Combines CSV and JSON files
- Removes duplicates
- Performs data validation and diagnostics
- Generates combined CSV output
- Creates analysis-ready datasets

**Required Data Files**:
- `data/viewblock_thorchain_*.csv` - THORChain CSV files
- `data/viewblock_thorchain_*.json` - THORChain JSON files

**Dependencies**:
- pandas, numpy, pathlib, glob, datetime

**Usage**:
1. Place all ViewBlock THORChain data files in `data/` directory
2. Open the notebook in Jupyter
3. Run all cells sequentially
4. Review the combined dataset and diagnostics
5. Output files will be saved with timestamps

---

### 4. Chainflip Volume Analyzer (`chainflip_volume_analyzer.ipynb`)

**Purpose**: Combines multiple Chainflip CSV and JSON files, cleans the data, removes duplicates, and prepares it for volume analysis.

**Analysis Goals**:
- Average swap size
- Percentage basis points (bps) paid
- Assets swapped by day of week
- Trending assets week-to-week
- Trending assets month-to-month

**Required Data Files**:
- `data/swap_transactions_*.csv` - Chainflip swap transaction CSV files
- `data/swap_transactions_*.json` - Chainflip swap transaction JSON files (if any)

**Dependencies**:
- pandas, numpy, pathlib, glob, datetime, dateutil (optional)

**Usage**:
1. Place all Chainflip swap transaction files in `data/` directory
2. Open the notebook in Jupyter
3. Run all cells sequentially
4. Review the analysis results and visualizations

---

## Data Files

All data files should be placed in the `data/` directory. The notebooks are configured to read from this directory using relative paths.

### Data File Patterns:

- **TWAP Analysis**: `twaps.csv`, `foundation_twap_sales.csv`
- **THORChain**: `viewblock_thorchain_*.csv`, `viewblock_thorchain_*.json`
- **Chainflip**: `swap_transactions_*.csv`, `swap_transactions_*.json`
- **ButterSwap**: Transaction CSV files

## Configuration

### API Keys

Some notebooks may require API keys (e.g., CoinGecko API for price data). These should be configured within the notebook's configuration section.

**Note**: For security, consider using environment variables or a separate config file (not committed to git) for API keys.

## Output Files

Most notebooks generate output files with timestamps. By default, these are configured to save to the `data/` directory or a subdirectory. Check each notebook's output configuration for specific paths.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed via `pip install -r requirements.txt`

2. **FileNotFoundError**: Verify that data files are in the `data/` directory and filenames match expected patterns

3. **Path Issues**: Ensure you're running notebooks from the `notebooks/` directory or adjust paths accordingly

4. **Memory Issues**: Some notebooks process large datasets. If you encounter memory errors, consider:
   - Processing files in batches
   - Using a machine with more RAM
   - Optimizing data loading (chunking)

### Getting Help

If you encounter issues:
1. Check that all data files are present
2. Verify Python version compatibility (3.8+)
3. Ensure all dependencies are installed
4. Review notebook error messages for specific guidance

## Contributing

When adding new notebooks or modifying existing ones:
1. Update this README with notebook descriptions
2. Add any new dependencies to `requirements.txt`
3. Ensure data paths use relative paths from the `notebooks/` directory
4. Test notebooks with sample data

## License

[Specify your license here]

## Contact

[Add contact information if needed]




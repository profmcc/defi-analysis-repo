# Validation Cells Added

Validation cells have been added to all notebooks to automatically check data quality and catch errors.

## What Was Added

### 1. Validation Helper Imports
Each notebook now imports validation functions from `validation_helpers.py`:
- `validate_price_data()` - Price sanity checks
- `validate_amounts()` - Volume/amount validation
- `validate_fee_percentage()` - Fee reasonableness checks
- `validate_timestamps()` - Date range validation
- `validate_asset_pairs()` - Asset pair consistency
- `run_all_validations()` - Run all checks for a notebook type

### 2. Validation Cells Added

#### ButterSwap Data Analyzer
- ✅ After data cleaning (Step 4)
- ✅ After analysis dataframe creation (Step 5)

#### Chainflip Volume Analyzer
- ✅ After data combination (Step 2)
- ✅ After column mapping and cleaning

#### THORChain Data Combiner
- ✅ After data cleaning and deduplication (Step 4-5)

#### Master TWAP Threshold Analysis
- ✅ After market data loading
- ✅ After TWAP data combination

## Running Notebooks

### Option 1: Run All Notebooks Automatically
```bash
# Using Python script
python run_all_notebooks.py

# Or using shell script (double-click on Mac/Linux)
./run_all.sh
```

### Option 2: Run Individual Notebooks
Open each notebook in Jupyter and run all cells sequentially.

## What Validations Check

### Price Data
- ✅ Positive values
- ✅ Within expected range ($0.01 - $1000 for FOX)
- ✅ Reasonable volatility (CV < 200%)
- ✅ No sudden jumps (>50% daily change)

### Volumes/Amounts
- ✅ Positive (or non-negative for outputs)
- ✅ Reasonable magnitudes
- ✅ No extreme outliers

### Fees
- ✅ Non-negative
- ✅ Reasonable percentage (0.1% - 0.5% typical)
- ✅ Never exceeds swap amount

### Aggregations
- ✅ Sum = mean × count (within 1%)
- ✅ Combined = sum of parts

### Timestamps
- ✅ Reasonable date range (2020-2030)
- ✅ Not all duplicates

## Validation Output

When validations run, you'll see:
- ✓ All validations passed! (if everything is good)
- ⚠ Found X validation issue(s): (if issues found)
  - List of specific issues to review

## Files Created

1. **`validation_helpers.py`** - Reusable validation functions
2. **`VALIDATION_FRAMEWORK.md`** - Comprehensive validation documentation
3. **`VALIDATION_QUICK_REFERENCE.md`** - Quick reference guide
4. **`run_all_notebooks.py`** - Python script to run all notebooks
5. **`run_all.sh`** - Shell script for easy execution

## Next Steps

1. Run the notebooks using `python run_all_notebooks.py`
2. Review any validation warnings
3. Fix data issues if found
4. Re-run validations to confirm fixes

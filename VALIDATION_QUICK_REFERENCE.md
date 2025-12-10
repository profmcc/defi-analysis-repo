# Validation Quick Reference

Quick guide for validating values in DeFi analysis notebooks.

## Quick Checks by Notebook

### Master TWAP Analysis
```python
# After loading price data
✓ FOX price: $0.01 - $10 (warn if < $0.001 or > $100)
✓ Volume: $1K - $100M daily (warn if < $100 or > $1B)
✓ Daily returns: -50% to +50% (warn if < -90% or > +90%)

# After loading TWAPs
✓ TWAP amounts: 1K - 10M FOX per day (warn if < 100 or > 100M)
✓ Combined TWAPs = DAO TWAPs + Foundation TWAPs (within 0.01 FOX)

# After regression
✓ Coefficient: |coef| < 1e-6 (1M FOX should move price < 1%)
✓ R²: 0.01 - 0.95 (warn if > 0.99 = overfitting)
✓ P-value: 0 - 1 (warn if < 0.001 = verify data quality)
```

### ButterSwap Analyzer
```python
# After loading swaps
✓ Swap size (USD): $10 - $100K median (warn if < $1 or > $10M)
✓ Fee percentage: 0.1% - 0.5% median (warn if < 0.01% or > 5%)
✓ Fee < swap amount (never exceeds 10% of swap)

# After aggregations
✓ Total volume = mean * count (within 1%)
✓ Median < mean * 2 (typical for financial data)
```

### Chainflip Analyzer
```python
# After loading swaps
✓ Swap size (USD): Similar to ButterSwap
✓ Commission: 0.1% - 0.3% (warn if < 0.05% or > 5%)
✓ Age: < 365 days (warn if > 1 year old)

# Intermediate swaps
✓ Intermediate value: 50% - 200% of input value
```

### THORChain Combiner
```python
# After combining data
✓ Swap amounts: Positive (warn if negative)
✓ Asset pairs: from_asset ≠ to_asset (warn if same)
✓ Block height: Positive (warn if <= 0)
✓ Fee rate: 10-50% of swaps have fees (warn if < 1% or > 80%)
```

## Common Validation Patterns

### 1. Price Data
- ✅ Positive values
- ✅ Within expected range (e.g., $0.01 - $1000)
- ✅ Reasonable volatility (CV < 200%)
- ✅ No sudden jumps (>50% daily change)

### 2. Volume/Amounts
- ✅ Positive (or non-negative for outputs)
- ✅ Reasonable magnitudes
- ✅ No extreme outliers (99.9th / 0.1th percentile ratio < 10,000)

### 3. Fees
- ✅ Non-negative
- ✅ Reasonable percentage (0.1% - 0.5% typical)
- ✅ Never exceeds swap amount
- ✅ Median fee in expected range

### 4. Aggregations
- ✅ Sum = mean × count (within 1%)
- ✅ Combined = sum of parts (within 0.01)
- ✅ Median < mean × 2 (typical distribution)

### 5. Timestamps
- ✅ Reasonable date range (2020-2030)
- ✅ Not all duplicates (< 50% duplicate rate)
- ✅ Chronological order (if expected)

## Using Validation Helpers

### Basic Usage
```python
from validation_helpers import validate_price_data, validate_amounts

# Validate price column
issues = validate_price_data(df, 'price_fox')
if issues:
    for issue in issues:
        print(f"⚠ {issue}")
```

### Run All Validations
```python
from validation_helpers import run_all_validations

# Automatically runs relevant checks for notebook type
results = run_all_validations(df, notebook_type='butterswap', verbose=True)
```

### Custom Validation
```python
from validation_helpers import validate_fee_percentage

# Check fees are reasonable
fee_issues = validate_fee_percentage(
    df, 
    fee_col='fee_usd', 
    amount_col='from_amount_usd',
    min_fee_pct=0.05,  # Minimum 0.05%
    max_fee_pct=5.0    # Maximum 5%
)
```

## Expected Value Ranges

| Metric | Typical Range | Warning Threshold |
|--------|--------------|-------------------|
| **FOX Price** | $0.01 - $10 | < $0.001 or > $100 |
| **Daily Volume** | $1K - $100M | < $100 or > $1B |
| **Swap Size** | $10 - $100K | < $1 or > $10M |
| **Fee %** | 0.1% - 0.5% | < 0.01% or > 5% |
| **Daily Return** | -50% to +50% | < -90% or > +90% |
| **TWAP Amount** | 1K - 10M FOX | < 100 or > 100M FOX |
| **R²** | 0.01 - 0.95 | > 0.99 (overfitting) |

## Red Flags to Watch For

🚩 **Prices**
- Negative or zero prices
- Prices > 100x expected range
- Daily changes > 100%

🚩 **Volumes**
- Negative volumes
- Volumes > 1000x expected
- Zero volumes on active days

🚩 **Fees**
- Fees > 10% of swap amount
- Negative fees
- Fees = 0 when expected

🚩 **Aggregations**
- Sum ≠ mean × count
- Combined ≠ sum of parts
- Median > mean × 2

🚩 **Correlations**
- Correlation > 0.95 (suspiciously high)
- Correlation < -0.5 (suspiciously negative)

🚩 **Regression**
- R² > 0.99 (overfitting)
- Coefficients > 1e-6 (unrealistic impact)
- P-value < 0.001 (verify data quality)

## Quick Validation Checklist

Before trusting any calculated value, verify:

- [ ] Value is positive (or non-negative if appropriate)
- [ ] Value is within expected range
- [ ] No extreme outliers (> 99.9th percentile)
- [ ] Related values are consistent (sums match, etc.)
- [ ] Timestamps are reasonable
- [ ] Fees are reasonable percentages
- [ ] Aggregations are mathematically consistent
- [ ] Correlations are not suspiciously high/low

## Integration Example

Add this cell after key calculations in your notebooks:

```python
# Validation cell
from validation_helpers import run_all_validations

# Replace 'butterswap' with your notebook type: 'twap', 'chainflip', 'thorchain'
validation_results = run_all_validations(df, notebook_type='butterswap', verbose=True)

# Check if any critical issues
critical_issues = sum(len(issues) for issues in validation_results.values())
if critical_issues > 0:
    print(f"\n⚠ Found {critical_issues} validation issue(s) - review before proceeding")
else:
    print("\n✓ All validations passed - data looks good!")
```

## Files

- **`VALIDATION_FRAMEWORK.md`**: Comprehensive validation documentation
- **`validation_helpers.py`**: Reusable validation functions
- **`VALIDATION_QUICK_REFERENCE.md`**: This file (quick reference)

# DeFi Analysis Repository - Validation Framework

This document outlines validation checks and "sniff tests" to ensure calculated values are reasonable and accurate across all notebooks.

## Overview

Each notebook performs different types of calculations that need validation:
- **Price data** (TWAP analysis)
- **Volume calculations** (swap sizes, total volumes)
- **Fee calculations** (basis points, percentages)
- **Statistical aggregations** (means, medians, sums)
- **Regression coefficients** (price impact analysis)

---

## 1. Master TWAP Threshold Analysis (`master_twap_threshold_analysis.ipynb`)

### Values to Validate

#### 1.1 Price Data (CoinGecko API)
**Values:**
- `price_fox`, `price_eth`, `price_btc`
- `vol_fox` (trading volume)

**Validation Checks:**
```python
# Price sanity checks
assert df['price_fox'].min() > 0, "FOX price must be positive"
assert df['price_fox'].max() < 1000, "FOX price suspiciously high (>$1000)"
assert df['price_fox'].mean() > 0.01, "FOX price suspiciously low (<$0.01)"
assert df['price_fox'].std() / df['price_fox'].mean() < 2.0, "FOX price volatility too high (CV > 200%)"

# Volume sanity checks
assert df['vol_fox'].min() >= 0, "Volume cannot be negative"
assert df['vol_fox'].max() < 1e12, "Volume suspiciously high (>$1T)"
assert df['vol_fox'].median() > 0, "Most days should have non-zero volume"

# Price correlation checks
fox_eth_corr = df[['price_fox', 'price_eth']].corr().iloc[0,1]
assert abs(fox_eth_corr) < 0.95, f"FOX-ETH correlation suspiciously high ({fox_eth_corr:.2f})"
```

#### 1.2 TWAP Amounts
**Values:**
- `twaps_amount` (DAO TWAPs in FOX tokens)
- `foundation_twaps_fox` (Foundation TWAPs in FOX tokens)
- `combined_twaps_fox` (sum of both)

**Validation Checks:**
```python
# TWAP amount sanity checks
assert df['twaps_amount'].min() >= 0, "TWAP amounts cannot be negative"
assert df['twaps_amount'].max() < 1e9, "Single-day TWAP suspiciously high (>1B FOX)"
assert df['twaps_amount'].sum() > 0, "Total TWAPs must be positive"

# Foundation TWAP checks
assert df['foundation_twaps_fox'].min() >= 0, "Foundation TWAPs cannot be negative"
assert df['foundation_twaps_fox'].max() < 1e9, "Single-day Foundation TWAP suspiciously high"

# Combined TWAP checks
combined_total = df['combined_twaps_fox'].sum()
dao_total = df['twaps_amount'].sum()
foundation_total = df['foundation_twaps_fox'].sum()
assert abs(combined_total - (dao_total + foundation_total)) < 0.01, \
    f"Combined TWAPs ({combined_total}) != DAO ({dao_total}) + Foundation ({foundation_total})"

# Daily TWAP vs market cap check
daily_twap_pct = (df['combined_twaps_fox'] / (df['price_fox'] * 1e9)).max()  # Assuming ~1B supply
assert daily_twap_pct < 0.10, f"Daily TWAP exceeds 10% of market cap ({daily_twap_pct*100:.2f}%)"
```

#### 1.3 Volume Normalization
**Values:**
- `vol7d` (7-day rolling average volume)
- `vol_norm` (normalized volume)

**Validation Checks:**
```python
# Volume normalization checks
mean_vol = df['vol7d'].mean()
std_vol = df['vol7d'].std()
assert std_vol > 0, "Volume standard deviation must be positive"
assert df['vol_norm'].mean() < 0.1, "Normalized volume mean should be near zero"
assert df['vol_norm'].std() > 0.5, "Normalized volume should have reasonable spread"
assert df['vol_norm'].abs().max() < 5, "Normalized volume outliers should be < 5 std devs"
```

#### 1.4 Price Returns
**Values:**
- `ret_fox`, `ret_eth`, `ret_btc` (daily returns)

**Validation Checks:**
```python
# Return sanity checks
assert df['ret_fox'].abs().max() < 1.0, "Daily return exceeds 100% (likely error)"
assert df['ret_fox'].std() < 0.5, "Daily return volatility suspiciously high (>50%)"
assert df['ret_fox'].mean() > -0.1 and df['ret_fox'].mean() < 0.1, \
    f"Average daily return suspicious ({df['ret_fox'].mean():.2%})"

# Return correlation checks
ret_corr = df[['ret_fox', 'ret_eth', 'ret_btc']].corr()
assert ret_corr.loc['ret_fox', 'ret_eth'] > -0.5, "FOX-ETH return correlation suspiciously negative"
assert ret_corr.loc['ret_fox', 'ret_eth'] < 0.9, "FOX-ETH return correlation suspiciously high"
```

#### 1.5 Regression Coefficients
**Values:**
- Regression coefficients for TWAP impact on price
- P-values, R-squared values

**Validation Checks:**
```python
# Regression coefficient sanity checks
assert abs(coef_twap) < 1e-6, f"TWAP coefficient suspiciously large ({coef_twap:.2e})"
# Typical price impact: 1M FOX TWAP should move price < 1%
# If coef is in units of price change per FOX token:
# coef * 1e6 should be < 0.01 (1% price impact per 1M FOX)
assert abs(coef_twap * 1e6) < 0.01, \
    f"TWAP coefficient implies >1% price impact per 1M FOX ({coef_twap * 1e6:.4f})"

# R-squared checks
assert r_squared >= 0, "R-squared cannot be negative"
assert r_squared < 0.99, f"R-squared suspiciously high ({r_squared:.3f}) - possible overfitting"

# P-value checks
assert 0 <= p_value <= 1, "P-value must be between 0 and 1"
if p_value < 0.001:
    print(f"⚠ Very significant result (p={p_value:.4f}) - verify data quality")
```

---

## 2. ButterSwap Data Analyzer (`butterswap_data_analyzer.ipynb`)

### Values to Validate

#### 2.1 Swap Amounts
**Values:**
- `from_amount`, `to_amount` (raw token amounts)
- `from_amount_usd`, `to_amount_usd` (USD values)
- `swap_size` (primary swap size metric)

**Validation Checks:**
```python
# Amount sanity checks
assert df['from_amount'].min() > 0, "Swap amounts must be positive"
assert df['from_amount'].max() < 1e12, "Swap amount suspiciously high (>1T tokens)"
assert df['to_amount'].min() >= 0, "Output amounts cannot be negative"

# USD value checks
if 'from_amount_usd' in df.columns:
    assert df['from_amount_usd'].min() > 0, "USD values must be positive"
    assert df['from_amount_usd'].max() < 1e9, "Swap value suspiciously high (>$1B)"
    assert df['from_amount_usd'].median() > 10, "Median swap should be > $10"
    assert df['from_amount_usd'].median() < 1e6, "Median swap suspiciously high (>$1M)"

# Swap size consistency
if 'swap_size' in df.columns and 'from_amount_usd' in df.columns:
    size_diff = (df['swap_size'] - df['from_amount_usd']).abs().max()
    assert size_diff < 0.01, f"swap_size and from_amount_usd differ by > $0.01"
```

#### 2.2 Fee Calculations
**Values:**
- `fee_usd` (fee in USD)
- Fee percentage / basis points

**Validation Checks:**
```python
# Fee sanity checks
if 'fee_usd' in df.columns:
    assert df['fee_usd'].min() >= 0, "Fees cannot be negative"
    assert df['fee_usd'].max() < df['from_amount_usd'].max() * 0.1, \
        "Fee exceeds 10% of swap amount (likely error)"
    
    # Calculate fee percentage
    fee_pct = (df['fee_usd'] / df['from_amount_usd']) * 100
    assert fee_pct.min() >= 0, "Fee percentage cannot be negative"
    assert fee_pct.max() < 10, f"Fee percentage suspiciously high ({fee_pct.max():.2f}%)"
    assert fee_pct.median() < 1.0, f"Median fee suspiciously high ({fee_pct.median():.2f}%)"
    
    # Typical DeFi fees: 0.1% - 0.5%
    assert fee_pct.median() > 0.05, f"Median fee suspiciously low ({fee_pct.median():.2f}%)"
    assert fee_pct.median() < 2.0, f"Median fee suspiciously high ({fee_pct.median():.2f}%)"
```

#### 2.3 Statistical Aggregations
**Values:**
- Average swap size
- Total volume
- Median swap size

**Validation Checks:**
```python
# Aggregation sanity checks
avg_size = df['swap_size'].mean()
median_size = df['swap_size'].median()
total_volume = df['swap_size'].sum()

assert avg_size > 0, "Average swap size must be positive"
assert median_size > 0, "Median swap size must be positive"
assert total_volume > 0, "Total volume must be positive"

# Median should be less than mean (typical for financial data)
assert median_size < avg_size * 2, \
    f"Median ({median_size:.2f}) suspiciously close to mean ({avg_size:.2f})"

# Total volume should be reasonable
num_swaps = len(df)
expected_total = avg_size * num_swaps
assert abs(total_volume - expected_total) / expected_total < 0.01, \
    f"Total volume ({total_volume:.2f}) doesn't match avg * count ({expected_total:.2f})"
```

#### 2.4 Asset Pair Analysis
**Values:**
- Average swap size by asset pair
- Volume by asset pair

**Validation Checks:**
```python
# Asset pair checks
if 'asset_pair' in df.columns:
    pair_stats = df.groupby('asset_pair')['swap_size'].agg(['mean', 'count', 'sum'])
    
    # Each pair should have reasonable stats
    for pair, stats in pair_stats.iterrows():
        assert stats['mean'] > 0, f"Pair {pair} has non-positive average swap size"
        assert stats['count'] > 0, f"Pair {pair} has no swaps"
        assert stats['sum'] > 0, f"Pair {pair} has non-positive total volume"
        
        # Check for outliers
        pair_data = df[df['asset_pair'] == pair]['swap_size']
        if len(pair_data) > 10:
            q99 = pair_data.quantile(0.99)
            q01 = pair_data.quantile(0.01)
            assert q99 / q01 < 1000, \
                f"Pair {pair} has extreme outliers (99th/1st percentile ratio: {q99/q01:.1f})"
```

---

## 3. Chainflip Volume Analyzer (`chainflip_volume_analyzer.ipynb`)

### Values to Validate

#### 3.1 Swap Amounts (Similar to ButterSwap)
**Values:**
- `from_amount`, `to_amount`
- `from_amount_usd`, `to_amount_usd`
- `input_amount`, `output_amount` (Chainflip-specific)

**Validation Checks:**
```python
# Similar checks to ButterSwap
assert df['from_amount'].min() > 0, "Input amounts must be positive"
assert df['to_amount'].min() >= 0, "Output amounts cannot be negative"

# USD value consistency
if 'from_amount_usd' in df.columns and 'input_usd_value' in df.columns:
    usd_diff = (df['from_amount_usd'] - df['input_usd_value']).abs().max()
    assert usd_diff < 0.01, \
        f"from_amount_usd and input_usd_value differ by > $0.01 (max diff: {usd_diff:.2f})"
```

#### 3.2 Commission/Fee Validation
**Values:**
- `commission` / `fee_usd`
- Commission as percentage of swap

**Validation Checks:**
```python
# Commission sanity checks (Chainflip-specific)
if 'commission' in df.columns:
    assert df['commission'].min() >= 0, "Commission cannot be negative"
    
    # Chainflip fees are typically 0.1% - 0.3%
    if 'input_usd_value' in df.columns:
        comm_pct = (df['commission'] / df['input_usd_value']) * 100
        assert comm_pct.max() < 5.0, \
            f"Commission percentage suspiciously high ({comm_pct.max():.2f}%)"
        assert comm_pct.median() > 0.05, \
            f"Commission percentage suspiciously low ({comm_pct.median():.2f}%)"
        assert comm_pct.median() < 1.0, \
            f"Commission percentage suspiciously high ({comm_pct.median():.2f}%)"
```

#### 3.3 Intermediate Currency (Chainflip-specific)
**Values:**
- `intermediate_currency`, `intermediate_amount`
- `intermediate_usd_value`

**Validation Checks:**
```python
# Intermediate swap checks
if 'intermediate_amount' in df.columns:
    # For multi-hop swaps, intermediate amounts should be reasonable
    has_intermediate = df['intermediate_amount'].notna()
    if has_intermediate.sum() > 0:
        intermediate_data = df[has_intermediate]
        
        # Intermediate amount should be between input and output (roughly)
        # This is approximate - actual relationship depends on price ratios
        assert intermediate_data['intermediate_amount'].min() > 0, \
            "Intermediate amounts must be positive"
        
        # Intermediate USD value should be reasonable relative to input
        if 'intermediate_usd_value' in df.columns and 'input_usd_value' in df.columns:
            intermediate_ratio = (intermediate_data['intermediate_usd_value'] / 
                               intermediate_data['input_usd_value'])
            assert intermediate_ratio.min() > 0.5, \
                "Intermediate value suspiciously low (<50% of input)"
            assert intermediate_ratio.max() < 2.0, \
                "Intermediate value suspiciously high (>200% of input)"
```

#### 3.4 Age/Duration Metrics
**Values:**
- `age_minutes`, `age_hours`, `age_days`

**Validation Checks:**
```python
# Age consistency checks
if 'age_minutes' in df.columns and 'age_hours' in df.columns:
    hours_from_minutes = df['age_minutes'] / 60
    hour_diff = (df['age_hours'] - hours_from_minutes).abs().max()
    assert hour_diff < 0.1, \
        f"age_hours and age_minutes inconsistent (max diff: {hour_diff:.2f} hours)"

if 'age_hours' in df.columns and 'age_days' in df.columns:
    days_from_hours = df['age_hours'] / 24
    day_diff = (df['age_days'] - days_from_hours).abs().max()
    assert day_diff < 0.1, \
        f"age_days and age_hours inconsistent (max diff: {day_diff:.2f} days)"

# Age reasonableness (swaps shouldn't be years old in recent data)
if 'age_days' in df.columns:
    assert df['age_days'].max() < 365, \
        f"Swap age suspiciously high ({df['age_days'].max():.1f} days)"
    assert df['age_days'].min() >= 0, "Swap age cannot be negative"
```

---

## 4. THORChain Data Combiner (`thorchain_data_combiner.ipynb`)

### Values to Validate

#### 4.1 Swap Amounts
**Values:**
- `from_amount`, `to_amount`
- `swap_size`

**Validation Checks:**
```python
# Similar to other notebooks, but THORChain-specific
assert df['from_amount'].min() > 0, "THORChain swap amounts must be positive"
assert df['to_amount'].min() >= 0, "THORChain output amounts cannot be negative"

# THORChain swaps can be very small or very large
# But check for obvious errors
assert df['from_amount'].max() < 1e15, "Swap amount suspiciously high (>1e15)"
```

#### 4.2 Fee Detection
**Values:**
- `has_fee` (boolean)
- `affiliate_address` (indicates fee)

**Validation Checks:**
```python
# Fee detection checks
if 'has_fee' in df.columns:
    fee_rate = df['has_fee'].mean()
    # Not all swaps have fees, but some should
    assert fee_rate >= 0, "Fee rate cannot be negative"
    assert fee_rate <= 1, "Fee rate cannot exceed 100%"
    
    # Typical: 10-50% of swaps might have affiliate fees
    if fee_rate > 0.8:
        print(f"⚠ High fee rate ({fee_rate:.1%}) - verify fee detection logic")
    if fee_rate < 0.01:
        print(f"⚠ Low fee rate ({fee_rate:.1%}) - verify fee detection logic")
```

#### 4.3 Asset Name Validation
**Values:**
- `from_asset`, `to_asset`

**Validation Checks:**
```python
# Asset name checks
if 'from_asset' in df.columns:
    # Common THORChain assets
    common_assets = ['BTC', 'ETH', 'USDC', 'USDT', 'RUNE', 'BNB', 'AVAX']
    
    # Check for suspicious asset names
    unique_assets = df['from_asset'].unique()
    for asset in unique_assets:
        if pd.notna(asset):
            # Asset names should be reasonable length
            assert len(str(asset)) < 50, f"Asset name suspiciously long: {asset}"
            assert len(str(asset)) > 0, "Asset name cannot be empty"
            
            # Check for common typos or formatting issues
            if asset.upper() != asset and asset not in ['USDC', 'USDT']:
                print(f"⚠ Asset name has lowercase: {asset}")

# Asset pair consistency
if 'from_asset' in df.columns and 'to_asset' in df.columns:
    # Swaps shouldn't be from same asset to same asset
    same_asset_swaps = (df['from_asset'] == df['to_asset']).sum()
    if same_asset_swaps > 0:
        print(f"⚠ Found {same_asset_swaps} swaps from asset to itself - verify data")
```

#### 4.4 Block Height Validation
**Values:**
- `block_height`

**Validation Checks:**
```python
# Block height checks
if 'block_height' in df.columns:
    assert df['block_height'].min() > 0, "Block height must be positive"
    assert df['block_height'].is_monotonic_increasing or \
           df['block_height'].is_monotonic_decreasing or \
           len(df['block_height'].unique()) > 1, \
           "Block heights should vary (unless single block)"
    
    # Check for reasonable block height range
    # THORChain started around block 1, current blocks are in millions
    if df['block_height'].max() > 1e7:
        print(f"⚠ Block height suspiciously high: {df['block_height'].max():,.0f}")
```

---

## 5. Cross-Notebook Validation

### 5.1 Data Consistency Checks
```python
# If comparing across protocols, ensure consistent units
def validate_cross_protocol_consistency(df_butter, df_chainflip, df_thorchain):
    """Validate that metrics are comparable across protocols"""
    
    # Average swap sizes should be in similar ranges (USD)
    butter_avg = df_butter['swap_size'].mean() if 'swap_size' in df_butter.columns else None
    chainflip_avg = df_chainflip['swap_size'].mean() if 'swap_size' in df_chainflip.columns else None
    
    if butter_avg and chainflip_avg:
        ratio = max(butter_avg, chainflip_avg) / min(butter_avg, chainflip_avg)
        assert ratio < 100, \
            f"Average swap sizes differ by >100x (ButterSwap: ${butter_avg:.2f}, Chainflip: ${chainflip_avg:.2f})"
```

### 5.2 Temporal Consistency
```python
# Check that timestamps are reasonable
def validate_timestamps(df):
    """Validate timestamp data"""
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Timestamps should be in reasonable range
        min_date = timestamps.min()
        max_date = timestamps.max()
        
        assert min_date.year >= 2020, f"Earliest timestamp suspiciously old: {min_date}"
        assert max_date.year <= 2030, f"Latest timestamp suspiciously far in future: {max_date}"
        
        # Check for duplicate timestamps (might indicate data issues)
        duplicate_rate = timestamps.duplicated().mean()
        if duplicate_rate > 0.5:
            print(f"⚠ High duplicate timestamp rate ({duplicate_rate:.1%})")
```

---

## 6. Implementation: Validation Helper Functions

Create a shared validation module that can be imported by all notebooks:

```python
# validation_helpers.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

def validate_price_data(df: pd.DataFrame, price_col: str, 
                        min_price: float = 0.01, max_price: float = 1000) -> List[str]:
    """Validate price data and return list of warnings/errors"""
    issues = []
    
    if price_col not in df.columns:
        return [f"Column {price_col} not found"]
    
    prices = df[price_col]
    
    if prices.min() <= 0:
        issues.append(f"{price_col}: Found non-positive prices")
    if prices.max() > max_price:
        issues.append(f"{price_col}: Max price suspiciously high (${prices.max():.2f})")
    if prices.mean() < min_price:
        issues.append(f"{price_col}: Mean price suspiciously low (${prices.mean():.4f})")
    
    cv = prices.std() / prices.mean()
    if cv > 2.0:
        issues.append(f"{price_col}: High volatility (CV={cv:.2f})")
    
    return issues

def validate_amounts(df: pd.DataFrame, amount_col: str, 
                     min_amount: float = 0, max_amount: float = 1e12) -> List[str]:
    """Validate amount/volume data"""
    issues = []
    
    if amount_col not in df.columns:
        return [f"Column {amount_col} not found"]
    
    amounts = df[amount_col]
    
    if amounts.min() < min_amount:
        issues.append(f"{amount_col}: Found negative amounts")
    if amounts.max() > max_amount:
        issues.append(f"{amount_col}: Max amount suspiciously high ({amounts.max():.2e})")
    
    return issues

def validate_fee_percentage(df: pd.DataFrame, fee_col: str, amount_col: str,
                            min_fee_pct: float = 0.05, max_fee_pct: float = 5.0) -> List[str]:
    """Validate fee as percentage of amount"""
    issues = []
    
    if fee_col not in df.columns or amount_col not in df.columns:
        return [f"Required columns not found: {fee_col}, {amount_col}"]
    
    fee_pct = (df[fee_col] / df[amount_col]) * 100
    
    median_fee = fee_pct.median()
    if median_fee < min_fee_pct:
        issues.append(f"Fee percentage suspiciously low (median: {median_fee:.2f}%)")
    if median_fee > max_fee_pct:
        issues.append(f"Fee percentage suspiciously high (median: {median_fee:.2f}%)")
    
    max_fee = fee_pct.max()
    if max_fee > max_fee_pct * 2:
        issues.append(f"Fee percentage has extreme outliers (max: {max_fee:.2f}%)")
    
    return issues

def validate_regression_results(coef: float, r_squared: float, p_value: float,
                               coef_max: float = 1e-6) -> List[str]:
    """Validate regression results"""
    issues = []
    
    if abs(coef) > coef_max:
        issues.append(f"Coefficient suspiciously large ({coef:.2e})")
    
    if r_squared < 0 or r_squared > 1:
        issues.append(f"R-squared out of valid range ({r_squared:.3f})")
    if r_squared > 0.99:
        issues.append(f"R-squared suspiciously high ({r_squared:.3f}) - possible overfitting")
    
    if p_value < 0 or p_value > 1:
        issues.append(f"P-value out of valid range ({p_value:.4f})")
    
    return issues

def run_all_validations(df: pd.DataFrame, notebook_type: str) -> Dict[str, List[str]]:
    """Run all relevant validations for a notebook type"""
    results = {}
    
    if notebook_type == "twap":
        if 'price_fox' in df.columns:
            results['price_fox'] = validate_price_data(df, 'price_fox')
        if 'vol_fox' in df.columns:
            results['vol_fox'] = validate_amounts(df, 'vol_fox', min_amount=0, max_amount=1e12)
    
    elif notebook_type == "butterswap":
        if 'from_amount_usd' in df.columns:
            results['amounts'] = validate_amounts(df, 'from_amount_usd', max_amount=1e9)
        if 'fee_usd' in df.columns and 'from_amount_usd' in df.columns:
            results['fees'] = validate_fee_percentage(df, 'fee_usd', 'from_amount_usd')
    
    # Add more notebook-specific validations...
    
    return results
```

---

## 7. Usage Recommendations

### 7.1 Add Validation Cells to Notebooks
Add validation cells after key calculations:

```python
# After loading price data
price_issues = validate_price_data(df, 'price_fox')
if price_issues:
    print("⚠ Price data validation issues:")
    for issue in price_issues:
        print(f"  - {issue}")
else:
    print("✓ Price data passed validation")

# After calculating fees
if 'fee_usd' in df.columns:
    fee_issues = validate_fee_percentage(df, 'fee_usd', 'from_amount_usd')
    if fee_issues:
        print("⚠ Fee validation issues:")
        for issue in fee_issues:
            print(f"  - {issue}")
```

### 7.2 Automated Validation Script
Create a script that runs all notebooks and validates outputs:

```python
# validate_all_notebooks.py
import subprocess
import json
from pathlib import Path

def validate_notebook(notebook_path: Path) -> Dict:
    """Run notebook and collect validation results"""
    # Execute notebook
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--execute', '--to', 'notebook', str(notebook_path)],
        capture_output=True
    )
    
    # Parse outputs and run validations
    # (Implementation depends on how notebooks output data)
    
    return {'notebook': str(notebook_path), 'status': 'success', 'issues': []}
```

### 7.3 Pre-commit Hooks
Add validation checks to git pre-commit hooks to catch issues before committing.

---

## 8. Summary Checklist

Before trusting any calculated values, verify:

- [ ] **Prices**: Within expected ranges, positive, reasonable volatility
- [ ] **Volumes**: Positive, reasonable magnitudes, consistent with expectations
- [ ] **Amounts**: Positive (or non-negative for outputs), reasonable magnitudes
- [ ] **Fees**: Reasonable percentages (typically 0.1% - 0.5%), not exceeding swap amounts
- [ ] **Aggregations**: Sums match component sums, means/medians reasonable
- [ ] **Correlations**: Not suspiciously high/low (indicates data issues)
- [ ] **Regression**: Coefficients reasonable, R² not overfitted, p-values valid
- [ ] **Timestamps**: In reasonable date ranges, not all duplicates
- [ ] **Cross-checks**: Related values are consistent (e.g., combined = sum of parts)
- [ ] **Outliers**: Extreme values investigated and explained

---

## 9. Quick Reference: Expected Ranges

| Metric | Expected Range | Warning Threshold |
|--------|---------------|-------------------|
| FOX Price | $0.01 - $10 | < $0.001 or > $100 |
| Daily Volume | $1K - $100M | < $100 or > $1B |
| Swap Size (USD) | $10 - $100K | < $1 or > $10M |
| Fee Percentage | 0.1% - 0.5% | < 0.01% or > 5% |
| Daily Return | -50% to +50% | < -90% or > +90% |
| TWAP Amount | 1K - 10M FOX | < 100 or > 100M FOX |
| R-squared | 0 - 0.95 | > 0.99 (overfitting) |

---

This validation framework should be applied consistently across all notebooks to ensure data quality and catch errors early.

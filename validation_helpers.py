"""
Validation helpers for DeFi analysis notebooks.

This module provides reusable validation functions to check that calculated
values pass "sniff tests" and are within expected ranges.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def validate_price_data(
    df: pd.DataFrame,
    price_col: str,
    min_price: float = 0.01,
    max_price: float = 1000,
    max_cv: float = 2.0
) -> List[str]:
    """
    Validate price data and return list of warnings/errors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    price_col : str
        Name of price column to validate
    min_price : float
        Minimum reasonable price (default: $0.01)
    max_price : float
        Maximum reasonable price (default: $1000)
    max_cv : float
        Maximum coefficient of variation (default: 2.0 = 200%)
    
    Returns:
    --------
    List[str]
        List of validation issues (empty if all checks pass)
    """
    issues = []
    
    if price_col not in df.columns:
        return [f"Column '{price_col}' not found"]
    
    prices = df[price_col].dropna()
    
    if len(prices) == 0:
        return [f"Column '{price_col}' has no valid data"]
    
    if prices.min() <= 0:
        issues.append(f"{price_col}: Found non-positive prices (min: {prices.min():.6f})")
    
    if prices.max() > max_price:
        issues.append(f"{price_col}: Max price suspiciously high (${prices.max():.2f} > ${max_price})")
    
    if prices.mean() < min_price:
        issues.append(f"{price_col}: Mean price suspiciously low (${prices.mean():.4f} < ${min_price})")
    
    cv = prices.std() / prices.mean() if prices.mean() > 0 else np.inf
    if cv > max_cv:
        issues.append(f"{price_col}: High volatility (CV={cv:.2f} > {max_cv})")
    
    # Check for sudden jumps (>50% change)
    if len(prices) > 1:
        pct_change = prices.pct_change().abs()
        large_jumps = (pct_change > 0.5).sum()
        if large_jumps > len(prices) * 0.1:  # More than 10% of days
            issues.append(f"{price_col}: Many large price jumps ({large_jumps} days with >50% change)")
    
    return issues


def validate_amounts(
    df: pd.DataFrame,
    amount_col: str,
    min_amount: float = 0,
    max_amount: float = 1e12,
    allow_zero: bool = False
) -> List[str]:
    """
    Validate amount/volume data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing amount data
    amount_col : str
        Name of amount column to validate
    min_amount : float
        Minimum reasonable amount (default: 0)
    max_amount : float
        Maximum reasonable amount (default: 1e12)
    allow_zero : bool
        Whether zero amounts are allowed (default: False)
    
    Returns:
    --------
    List[str]
        List of validation issues
    """
    issues = []
    
    if amount_col not in df.columns:
        return [f"Column '{amount_col}' not found"]
    
    amounts = df[amount_col].dropna()
    
    if len(amounts) == 0:
        return [f"Column '{amount_col}' has no valid data"]
    
    # Convert to numeric if needed (handles string numbers)
    if amounts.dtype == 'object':
        amounts = pd.to_numeric(amounts, errors='coerce').dropna()
        if len(amounts) == 0:
            return [f"Column '{amount_col}' contains no numeric data"]
    
    if not allow_zero and amounts.min() <= 0:
        issues.append(f"{amount_col}: Found non-positive amounts (min: {amounts.min():.6f})")
    elif allow_zero and amounts.min() < 0:
        issues.append(f"{amount_col}: Found negative amounts (min: {amounts.min():.6f})")
    
    if amounts.max() > max_amount:
        issues.append(f"{amount_col}: Max amount suspiciously high ({amounts.max():.2e} > {max_amount:.2e})")
    
    # Check for extreme outliers (beyond 99.9th percentile)
    # Only flag if ratio is extremely high AND both percentiles are meaningful
    if len(amounts) > 100:
        q999 = amounts.quantile(0.999)
        q001 = amounts.quantile(0.001)
        # Only flag if ratio is very high AND the lower percentile is meaningful (> 0.01)
        # This avoids false positives from very small amounts (like 0.000001) vs large amounts
        if q001 > 0.01 and q999 / q001 > 100000:
            issues.append(f"{amount_col}: Extreme outliers detected (99.9th/0.1th percentile ratio: {q999/q001:.1f})")
        elif q001 <= 0.01 and q999 / q001 > 100000:
            # Very small lower percentile - this is expected in financial data with wide ranges
            issues.append(f"{amount_col}: Wide value range (99.9th/0.1th percentile ratio: {q999/q001:.1f}) - this is normal for financial data")
    
    return issues


def validate_fee_percentage(
    df: pd.DataFrame,
    fee_col: str,
    amount_col: str,
    min_fee_pct: float = 0.05,
    max_fee_pct: float = 5.0,
    max_outlier_pct: float = 10.0
) -> List[str]:
    """
    Validate fee as percentage of amount.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing fee and amount data
    fee_col : str
        Name of fee column
    amount_col : str
        Name of amount column
    min_fee_pct : float
        Minimum reasonable fee percentage (default: 0.05%)
    max_fee_pct : float
        Maximum reasonable fee percentage (default: 5.0%)
    max_outlier_pct : float
        Maximum fee percentage for outliers (default: 10.0%)
    
    Returns:
    --------
    List[str]
        List of validation issues
    """
    issues = []
    
    if fee_col not in df.columns or amount_col not in df.columns:
        return [f"Required columns not found: '{fee_col}', '{amount_col}'"]
    
    # Filter to rows where both fee and amount are valid
    valid_mask = df[fee_col].notna() & df[amount_col].notna() & (df[amount_col] > 0)
    if valid_mask.sum() == 0:
        return [f"No valid fee/amount pairs found"]
    
    fee_pct = (df.loc[valid_mask, fee_col] / df.loc[valid_mask, amount_col]) * 100
    
    if fee_pct.min() < 0:
        issues.append(f"Fee percentage has negative values (min: {fee_pct.min():.2f}%)")
    
    median_fee = fee_pct.median()
    if median_fee < min_fee_pct:
        issues.append(f"Fee percentage suspiciously low (median: {median_fee:.2f}% < {min_fee_pct}%)")
    if median_fee > max_fee_pct:
        issues.append(f"Fee percentage suspiciously high (median: {median_fee:.2f}% > {max_fee_pct}%)")
    
    max_fee = fee_pct.max()
    if max_fee > max_outlier_pct:
        issues.append(f"Fee percentage has extreme outliers (max: {max_fee:.2f}% > {max_outlier_pct}%)")
    
    # Check for fees exceeding amount
    fee_exceeds_amount = (df.loc[valid_mask, fee_col] > df.loc[valid_mask, amount_col]).sum()
    if fee_exceeds_amount > 0:
        issues.append(f"{fee_exceeds_amount} rows have fees exceeding swap amount")
    
    return issues


def validate_regression_results(
    coef: float,
    r_squared: float,
    p_value: float,
    coef_max: float = 1e-6,
    coef_description: str = "coefficient"
) -> List[str]:
    """
    Validate regression results.
    
    Parameters:
    -----------
    coef : float
        Regression coefficient
    r_squared : float
        R-squared value
    p_value : float
        P-value
    coef_max : float
        Maximum reasonable coefficient magnitude (default: 1e-6)
    coef_description : str
        Description of coefficient for error messages
    
    Returns:
    --------
    List[str]
        List of validation issues
    """
    issues = []
    
    if abs(coef) > coef_max:
        issues.append(f"{coef_description} suspiciously large ({coef:.2e} > {coef_max:.2e})")
    
    if r_squared < 0 or r_squared > 1:
        issues.append(f"R-squared out of valid range ({r_squared:.3f})")
    elif r_squared > 0.99:
        issues.append(f"R-squared suspiciously high ({r_squared:.3f}) - possible overfitting")
    elif r_squared < 0.01:
        issues.append(f"R-squared suspiciously low ({r_squared:.3f}) - model may not be useful")
    
    if p_value < 0 or p_value > 1:
        issues.append(f"P-value out of valid range ({p_value:.4f})")
    elif p_value < 0.001:
        issues.append(f"⚠ Very significant result (p={p_value:.4f}) - verify data quality")
    
    return issues


def validate_returns(
    df: pd.DataFrame,
    return_col: str,
    max_daily_return: float = 1.0,
    max_volatility: float = 0.5
) -> List[str]:
    """
    Validate return/percentage change data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing return data
    return_col : str
        Name of return column
    max_daily_return : float
        Maximum reasonable daily return (default: 1.0 = 100%)
    max_volatility : float
        Maximum reasonable volatility (default: 0.5 = 50%)
    
    Returns:
    --------
    List[str]
        List of validation issues
    """
    issues = []
    
    if return_col not in df.columns:
        return [f"Column '{return_col}' not found"]
    
    returns = df[return_col].dropna()
    
    if len(returns) == 0:
        return [f"Column '{return_col}' has no valid data"]
    
    if returns.abs().max() > max_daily_return:
        issues.append(f"{return_col}: Daily return exceeds {max_daily_return*100:.0f}% (max: {returns.abs().max()*100:.2f}%)")
    
    volatility = returns.std()
    if volatility > max_volatility:
        issues.append(f"{return_col}: Volatility suspiciously high ({volatility*100:.2f}% > {max_volatility*100:.0f}%)")
    
    # Check for excessive consecutive moves in same direction
    if len(returns) > 10:
        sign_changes = (np.sign(returns) != np.sign(returns.shift(1))).sum()
        if sign_changes < len(returns) * 0.1:  # Less than 10% sign changes
            issues.append(f"{return_col}: Suspiciously few sign changes ({sign_changes}/{len(returns)})")
    
    return issues


def validate_aggregation_consistency(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None
) -> List[str]:
    """
    Validate that aggregations are consistent (e.g., sum of parts equals total).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing values
    value_col : str
        Name of value column
    group_col : str, optional
        If provided, check that sum of groups equals total
    
    Returns:
    --------
    List[str]
        List of validation issues
    """
    issues = []
    
    if value_col not in df.columns:
        return [f"Column '{value_col}' not found"]
    
    total = df[value_col].sum()
    
    if group_col and group_col in df.columns:
        group_sum = df.groupby(group_col)[value_col].sum().sum()
        if abs(total - group_sum) > 0.01:
            issues.append(f"Total ({total:.2f}) doesn't match sum of groups ({group_sum:.2f})")
    
    # Check that mean * count = sum
    mean = df[value_col].mean()
    count = len(df)
    expected_total = mean * count
    if abs(total - expected_total) / max(abs(total), 1) > 0.01:
        issues.append(f"Total ({total:.2f}) doesn't match mean * count ({expected_total:.2f})")
    
    return issues


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    min_year: int = 2020,
    max_year: int = 2030
) -> List[str]:
    """
    Validate timestamp data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing timestamps
    timestamp_col : str
        Name of timestamp column
    min_year : int
        Minimum reasonable year
    max_year : int
        Maximum reasonable year
    
    Returns:
    --------
    List[str]
        List of validation issues
    """
    issues = []
    
    if timestamp_col not in df.columns:
        return [f"Column '{timestamp_col}' not found"]
    
    try:
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        valid_timestamps = timestamps.dropna()
        
        if len(valid_timestamps) == 0:
            return [f"Column '{timestamp_col}' has no valid timestamps"]
        
        min_date = valid_timestamps.min()
        max_date = valid_timestamps.max()
        
        if min_date.year < min_year:
            issues.append(f"Earliest timestamp suspiciously old: {min_date} (year < {min_year})")
        
        if max_date.year > max_year:
            issues.append(f"Latest timestamp suspiciously far in future: {max_date} (year > {max_year})")
        
        # Check for duplicate timestamps
        duplicate_rate = valid_timestamps.duplicated().mean()
        if duplicate_rate > 0.5:
            issues.append(f"High duplicate timestamp rate ({duplicate_rate:.1%})")
        
        # Check for timestamps out of order (if that's unexpected)
        if len(valid_timestamps) > 1:
            is_sorted = valid_timestamps.is_monotonic_increasing or valid_timestamps.is_monotonic_decreasing
            if not is_sorted and duplicate_rate < 0.1:
                issues.append("⚠ Timestamps are not in chronological order")
        
    except Exception as e:
        issues.append(f"Error parsing timestamps: {e}")
    
    return issues


def validate_asset_pairs(
    df: pd.DataFrame,
    from_asset_col: str = 'from_asset',
    to_asset_col: str = 'to_asset'
) -> List[str]:
    """
    Validate asset pair data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing asset pair data
    from_asset_col : str
        Name of source asset column
    to_asset_col : str
        Name of destination asset column
    
    Returns:
    --------
    List[str]
        List of validation issues
    """
    issues = []
    
    if from_asset_col not in df.columns or to_asset_col not in df.columns:
        return [f"Required columns not found: '{from_asset_col}', '{to_asset_col}'"]
    
    # Check for swaps from same asset to same asset
    same_asset = (df[from_asset_col] == df[to_asset_col]).sum()
    if same_asset > 0:
        issues.append(f"Found {same_asset} swaps from asset to itself - verify data")
    
    # Check for missing asset names
    missing_from = df[from_asset_col].isna().sum()
    missing_to = df[to_asset_col].isna().sum()
    if missing_from > 0:
        issues.append(f"Found {missing_from} rows with missing source asset")
    if missing_to > 0:
        issues.append(f"Found {missing_to} rows with missing destination asset")
    
    # Check for suspiciously long asset names
    if df[from_asset_col].dtype == 'object':
        long_names = df[from_asset_col].astype(str).str.len() > 50
        if long_names.sum() > 0:
            issues.append(f"Found {long_names.sum()} rows with suspiciously long asset names")
    
    return issues


def run_all_validations(
    df: pd.DataFrame,
    notebook_type: str,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Run all relevant validations for a notebook type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    notebook_type : str
        Type of notebook: 'twap', 'butterswap', 'chainflip', 'thorchain'
    verbose : bool
        Whether to print validation results
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping validation category to list of issues
    """
    results = {}
    
    if notebook_type == "twap":
        if 'price_fox' in df.columns:
            results['price_fox'] = validate_price_data(df, 'price_fox')
        if 'vol_fox' in df.columns:
            results['vol_fox'] = validate_amounts(df, 'vol_fox', min_amount=0, max_amount=1e12)
        if 'ret_fox' in df.columns:
            results['returns'] = validate_returns(df, 'ret_fox')
        if 'combined_twaps_fox' in df.columns:
            results['twaps'] = validate_amounts(df, 'combined_twaps_fox', max_amount=1e9)
        if 'timestamp' in df.columns:
            results['timestamps'] = validate_timestamps(df)
    
    elif notebook_type == "butterswap":
        if 'from_amount_usd' in df.columns:
            results['amounts'] = validate_amounts(df, 'from_amount_usd', max_amount=1e9)
        if 'fee_usd' in df.columns and 'from_amount_usd' in df.columns:
            results['fees'] = validate_fee_percentage(df, 'fee_usd', 'from_amount_usd')
        if 'from_asset' in df.columns and 'to_asset' in df.columns:
            results['asset_pairs'] = validate_asset_pairs(df)
        if 'timestamp' in df.columns:
            results['timestamps'] = validate_timestamps(df)
    
    elif notebook_type == "chainflip":
        if 'from_amount_usd' in df.columns:
            results['amounts'] = validate_amounts(df, 'from_amount_usd', max_amount=1e9)
        if 'commission' in df.columns and 'input_usd_value' in df.columns:
            results['fees'] = validate_fee_percentage(df, 'commission', 'input_usd_value')
        if 'age_days' in df.columns:
            age_issues = []
            if df['age_days'].max() > 365:
                age_issues.append(f"Swap age suspiciously high ({df['age_days'].max():.1f} days)")
            results['age'] = age_issues
    
    elif notebook_type == "thorchain":
        if 'from_amount' in df.columns:
            results['amounts'] = validate_amounts(df, 'from_amount', max_amount=1e15)
        if 'from_asset' in df.columns and 'to_asset' in df.columns:
            results['asset_pairs'] = validate_asset_pairs(df)
        if 'block_height' in df.columns:
            bh_issues = []
            if df['block_height'].min() <= 0:
                bh_issues.append("Block height must be positive")
            results['block_height'] = bh_issues
    
    if verbose:
        print_validation_results(results)
    
    return results


def print_validation_results(results: Dict[str, List[str]]) -> None:
    """
    Print validation results in a readable format.
    
    Parameters:
    -----------
    results : Dict[str, List[str]]
        Dictionary mapping validation category to list of issues
    """
    total_issues = sum(len(issues) for issues in results.values())
    
    if total_issues == 0:
        print("✓ All validations passed!")
        return
    
    print(f"\n⚠ Found {total_issues} validation issue(s):\n")
    
    for category, issues in results.items():
        if issues:
            print(f"  {category.upper()}:")
            for issue in issues:
                print(f"    - {issue}")
            print()
    
    print("Please review these issues before trusting the results.")


# Example usage function
def example_usage():
    """Example of how to use validation functions in notebooks"""
    example_code = '''
# In your notebook, after loading/calculating data:

from validation_helpers import (
    validate_price_data,
    validate_amounts,
    validate_fee_percentage,
    run_all_validations,
    print_validation_results
)

# Option 1: Validate specific columns
price_issues = validate_price_data(df, 'price_fox')
if price_issues:
    print("⚠ Price data validation issues:")
    for issue in price_issues:
        print(f"  - {issue}")
else:
    print("✓ Price data passed validation")

# Option 2: Run all validations for notebook type
results = run_all_validations(df, notebook_type='butterswap', verbose=True)

# Option 3: Custom validation
fee_issues = validate_fee_percentage(df, 'fee_usd', 'from_amount_usd')
print_validation_results({'fees': fee_issues})
'''
    print(example_code)

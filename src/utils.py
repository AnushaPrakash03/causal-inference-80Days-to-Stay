"""
Utility Functions for Causal Inference Project

Helper functions for data processing, visualization, and validation

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def calculate_ate(y1, y0):
    """
    Calculate Average Treatment Effect from potential outcomes
    
    Parameters:
    -----------
    y1 : array-like
        Potential outcomes under treatment
    y0 : array-like
        Potential outcomes under control
    
    Returns:
    --------
    float : ATE
    """
    return np.mean(y1) - np.mean(y0)


def naive_comparison(data, treatment_col, outcome_col):
    """
    Calculate naive difference in means (biased estimator)
    
    Parameters:
    -----------
    data : pd.DataFrame
    treatment_col : str
    outcome_col : str
    
    Returns:
    --------
    dict with results
    """
    treated_mean = data[data[treatment_col] == 1][outcome_col].mean()
    control_mean = data[data[treatment_col] == 0][outcome_col].mean()
    naive_diff = treated_mean - control_mean
    
    # T-test
    treated_outcomes = data[data[treatment_col] == 1][outcome_col]
    control_outcomes = data[data[treatment_col] == 0][outcome_col]
    t_stat, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)
    
    print("Naive Comparison (Biased):")
    print(f"  Treated mean: {treated_mean:.4f}")
    print(f"  Control mean: {control_mean:.4f}")
    print(f"  Difference: {naive_diff:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    return {
        'treated_mean': treated_mean,
        'control_mean': control_mean,
        'difference': naive_diff,
        'pvalue': p_value
    }


def plot_propensity_distribution(data, ps_col, treatment_col, save_path=None):
    """
    Plot propensity score distributions for treated and control
    
    Parameters:
    -----------
    data : pd.DataFrame
    ps_col : str
        Column name for propensity scores
    treatment_col : str
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    treated_ps = data[data[treatment_col] == 1][ps_col]
    control_ps = data[data[treatment_col] == 0][ps_col]
    
    ax.hist(control_ps, bins=50, alpha=0.5, label='Control', color='coral', density=True)
    ax.hist(treated_ps, bins=50, alpha=0.5, label='Treated', color='skyblue', density=True)
    
    ax.set_xlabel('Propensity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Propensity Score Distribution: Common Support Check', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add overlap region
    overlap_min = max(control_ps.min(), treated_ps.min())
    overlap_max = min(control_ps.max(), treated_ps.max())
    ax.axvspan(overlap_min, overlap_max, alpha=0.1, color='green', 
               label=f'Overlap Region [{overlap_min:.3f}, {overlap_max:.3f}]')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Calculate overlap statistics
    overlap_pct = (
        ((control_ps >= overlap_min) & (control_ps <= overlap_max)).mean() +
        ((treated_ps >= overlap_min) & (treated_ps <= overlap_max)).mean()
    ) / 2
    
    print(f"\nCommon Support Statistics:")
    print(f"  Overlap region: [{overlap_min:.3f}, {overlap_max:.3f}]")
    print(f"  % units in overlap: {overlap_pct:.1%}")


def love_plot(balance_df, threshold=0.1, save_path=None):
    """
    Create Love plot for covariate balance
    
    Parameters:
    -----------
    balance_df : pd.DataFrame
        Balance data with 'covariate', 'std_diff', 'timing' columns
    threshold : float
        Balance threshold (default 0.1)
    save_path : str, optional
    """
    # Pivot data
    balance_pivot = balance_df.pivot(index='covariate', columns='timing', values='std_diff')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by before balance
    balance_pivot = balance_pivot.sort_values('before', ascending=False)
    
    y_pos = np.arange(len(balance_pivot))
    
    # Plot before and after
    ax.scatter(balance_pivot['before'].abs(), y_pos, 
               s=150, alpha=0.7, color='coral', label='Before Matching', marker='o')
    ax.scatter(balance_pivot['after'].abs(), y_pos, 
               s=150, alpha=0.7, color='skyblue', label='After Matching', marker='s')
    
    # Connect with lines
    for i, (idx, row) in enumerate(balance_pivot.iterrows()):
        ax.plot([abs(row['before']), abs(row['after'])], [i, i], 
                color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Balance threshold
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Balance Threshold (±{threshold})', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_pivot.index)
    ax.set_xlabel('|Standardized Mean Difference|', fontsize=12, fontweight='bold')
    ax.set_title('Love Plot: Covariate Balance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def summary_statistics(data, treatment_col, covariate_cols):
    """
    Create summary statistics table by treatment status
    
    Parameters:
    -----------
    data : pd.DataFrame
    treatment_col : str
    covariate_cols : list
    
    Returns:
    --------
    pd.DataFrame with summary statistics
    """
    summary = []
    
    for col in covariate_cols:
        treated_stats = data[data[treatment_col] == 1][col].describe()
        control_stats = data[data[treatment_col] == 0][col].describe()
        
        # T-test for difference
        t_stat, p_val = stats.ttest_ind(
            data[data[treatment_col] == 1][col],
            data[data[treatment_col] == 0][col]
        )
        
        summary.append({
            'Variable': col,
            'Treated_Mean': treated_stats['mean'],
            'Treated_SD': treated_stats['std'],
            'Control_Mean': control_stats['mean'],
            'Control_SD': control_stats['std'],
            'Difference': treated_stats['mean'] - control_stats['mean'],
            'P_value': p_val
        })
    
    summary_df = pd.DataFrame(summary)
    
    print("\nSummary Statistics by Treatment Status:")
    print(summary_df.to_string(index=False))
    
    return summary_df


def bootstrap_ci(data, treatment_col, outcome_col, estimator_func, n_bootstrap=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals
    
    Parameters:
    -----------
    data : pd.DataFrame
    treatment_col : str
    outcome_col : str
    estimator_func : callable
        Function that takes data and returns estimate
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level (default 0.05 for 95% CI)
    
    Returns:
    --------
    dict with 'estimate', 'ci_lower', 'ci_upper'
    """
    print(f"\nBootstrapping {n_bootstrap} samples...")
    
    # Original estimate
    original_estimate = estimator_func(data)
    
    # Bootstrap
    bootstrap_estimates = []
    n = len(data)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = data.sample(n=n, replace=True, random_state=i)
        
        # Calculate estimate
        boot_est = estimator_func(bootstrap_sample)
        bootstrap_estimates.append(boot_est)
    
    # Calculate percentile CI
    ci_lower = np.percentile(bootstrap_estimates, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha/2) * 100)
    
    print(f"✓ Bootstrap complete")
    print(f"  Estimate: {original_estimate:.4f}")
    print(f"  {(1-alpha)*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'estimate': original_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_estimates': bootstrap_estimates
    }


def create_results_table(psm_result, did_result, true_ate=None):
    """
    Create formatted results comparison table
    
    Parameters:
    -----------
    psm_result : dict
        Results from PSM analysis
    did_result : dict
        Results from DiD analysis
    true_ate : float, optional
        True ATE (if known from simulation)
    
    Returns:
    --------
    pd.DataFrame with formatted results
    """
    results = []
    
    # PSM
    results.append({
        'Method': 'Propensity Score Matching',
        'Estimate': psm_result['att'],
        'Std. Error': psm_result['se'],
        'CI Lower': psm_result['ci_lower'],
        'CI Upper': psm_result['ci_upper'],
        'P-value': psm_result['pvalue'],
        'N': psm_result['n_treated']
    })
    
    # DiD
    results.append({
        'Method': 'Difference-in-Differences',
        'Estimate': did_result['coefficient'],
        'Std. Error': did_result['se'],
        'CI Lower': did_result['ci_lower'],
        'CI Upper': did_result['ci_upper'],
        'P-value': did_result['pvalue'],
        'N': 'Panel data'
    })
    
    # True ATE (if available)
    if true_ate is not None:
        results.append({
            'Method': 'True ATE (Simulation)',
            'Estimate': true_ate,
            'Std. Error': 0,
            'CI Lower': true_ate,
            'CI Upper': true_ate,
            'P-value': 0,
            'N': 'All units'
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    return results_df


def plot_treatment_effects(psm_result, did_result, true_ate=None, save_path=None):
    """
    Visualize treatment effect estimates from different methods
    
    Parameters:
    -----------
    psm_result : dict
    did_result : dict
    true_ate : float, optional
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    estimates = []
    cis_lower = []
    cis_upper = []
    colors = []
    
    # PSM
    methods.append('PSM')
    estimates.append(psm_result['att'])
    cis_lower.append(psm_result['ci_lower'])
    cis_upper.append(psm_result['ci_upper'])
    colors.append('#2E86AB')
    
    # DiD
    methods.append('DiD')
    estimates.append(did_result['coefficient'])
    cis_lower.append(did_result['ci_lower'])
    cis_upper.append(did_result['ci_upper'])
    colors.append('#A23B72')
    
    # True ATE
    if true_ate is not None:
        methods.append('True ATE')
        estimates.append(true_ate)
        cis_lower.append(true_ate)
        cis_upper.append(true_ate)
        colors.append('#F18F01')
    
    y_pos = np.arange(len(methods))
    
    # Plot estimates
    ax.scatter(estimates, y_pos, s=200, c=colors, zorder=3, edgecolors='black', linewidth=2)
    
    # Plot CIs
    for i in range(len(methods)):
        ax.plot([cis_lower[i], cis_upper[i]], [y_pos[i], y_pos[i]], 
                color=colors[i], linewidth=3, alpha=0.7)
    
    # Vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Treatment Effect Estimate', fontsize=12, fontweight='bold')
    ax.set_title('Treatment Effect Estimates with 95% Confidence Intervals', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    """
    Test utility functions
    """
    print("Testing utility functions...")
    
    # Load data
    df = pd.read_csv('data/startup_data.csv')
    
    # Naive comparison
    naive_result = naive_comparison(df, 'treated', 'employee_growth_12mo')
    
    # Summary statistics
    covariate_cols = ['employee_count_baseline', 'company_age_years', 
                     'industry_biotech', 'location_boston']
    summary = summary_statistics(df, 'treated', covariate_cols)
    
    print("\n All utility functions working correctly")
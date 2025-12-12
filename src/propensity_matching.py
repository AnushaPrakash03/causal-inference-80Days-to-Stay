"""
Propensity Score Matching Implementation

Implements PSM for estimating Average Treatment Effect on Treated (ATT)

Key Steps:
1. Estimate propensity scores via logistic regression
2. Match treated to control units (nearest neighbor)
3. Check covariate balance
4. Estimate treatment effect

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns


class PropensityScoreMatcher:
    """
    Propensity Score Matching for causal inference
    """
    
    def __init__(self, treatment_col, outcome_col, covariate_cols):
        """
        Initialize PSM estimator
        
        Parameters:
        -----------
        treatment_col : str
            Column name for treatment indicator (binary)
        outcome_col : str
            Column name for outcome variable
        covariate_cols : list of str
            Column names for covariates used in PS estimation
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariate_cols = covariate_cols
        self.ps_model = None
        self.propensity_scores = None
        
    def fit(self, data):
        """
        Estimate propensity scores
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with treatment, outcome, and covariates
        """
        print("Step 1: Estimating propensity scores...")
        
        X = data[self.covariate_cols]
        y = data[self.treatment_col]
        
        # Fit logistic regression
        self.ps_model = LogisticRegression(random_state=42, max_iter=1000)
        self.ps_model.fit(X, y)
        
        # Predict propensity scores
        self.propensity_scores = self.ps_model.predict_proba(X)[:, 1]
        
        print(f"✓ Estimated propensity scores")
        print(f"  - Treated range: [{self.propensity_scores[y==1].min():.3f}, "
              f"{self.propensity_scores[y==1].max():.3f}]")
        print(f"  - Control range: [{self.propensity_scores[y==0].min():.3f}, "
              f"{self.propensity_scores[y==0].max():.3f}]")
        
        return self
    
    def match(self, data, caliper=0.01, replace=False):
        """
        Perform nearest neighbor matching
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with propensity scores
        caliper : float
            Maximum allowed distance in propensity scores
        replace : bool
            Whether to allow matching with replacement
        
        Returns:
        --------
        pd.DataFrame with columns: treated_idx, control_idx, ps_distance
        """
        print(f"\nStep 2: Matching (caliper={caliper}, replacement={replace})...")
        
        # Add PS to data
        data_with_ps = data.copy()
        data_with_ps['propensity_score'] = self.propensity_scores
        
        treated = data_with_ps[data_with_ps[self.treatment_col] == 1].copy()
        control = data_with_ps[data_with_ps[self.treatment_col] == 0].copy()
        
        matches = []
        used_controls = set()
        
        for treated_idx, treated_row in treated.iterrows():
            treated_ps = treated_row['propensity_score']
            
            # Calculate distances to all controls
            control_eligible = control if replace else control[~control.index.isin(used_controls)]
            
            if len(control_eligible) == 0:
                continue
            
            control_eligible['ps_distance'] = np.abs(
                control_eligible['propensity_score'] - treated_ps
            )
            
            # Apply caliper
            within_caliper = control_eligible[control_eligible['ps_distance'] <= caliper]
            
            if len(within_caliper) == 0:
                continue
            
            # Select nearest neighbor
            best_match_idx = within_caliper['ps_distance'].idxmin()
            best_distance = within_caliper.loc[best_match_idx, 'ps_distance']
            
            matches.append({
                'treated_idx': treated_idx,
                'control_idx': best_match_idx,
                'ps_distance': best_distance
            })
            
            if not replace:
                used_controls.add(best_match_idx)
        
        matches_df = pd.DataFrame(matches)
        
        print(f"✓ Matched {len(matches)} treated units ({len(matches)/len(treated):.1%} of treated)")
        print(f"  - Mean PS distance: {matches_df['ps_distance'].mean():.4f}")
        print(f"  - Max PS distance: {matches_df['ps_distance'].max():.4f}")
        
        return matches_df
    
    def check_balance(self, data, matches):
        """
        Check covariate balance before and after matching
        
        Parameters:
        -----------
        data : pd.DataFrame
        matches : pd.DataFrame from match()
        
        Returns:
        --------
        pd.DataFrame with balance statistics
        """
        print("\nStep 3: Checking covariate balance...")
        
        # Before matching
        treated = data[data[self.treatment_col] == 1]
        control = data[data[self.treatment_col] == 0]
        
        balance_before = self._calculate_balance(treated, control)
        balance_before['timing'] = 'before'
        
        # After matching
        treated_matched = data.loc[matches['treated_idx']]
        control_matched = data.loc[matches['control_idx']]
        
        balance_after = self._calculate_balance(treated_matched, control_matched)
        balance_after['timing'] = 'after'
        
        balance = pd.concat([balance_before, balance_after])
        
        print("\nCovariate Balance (Standardized Mean Differences):")
        print(balance.pivot(index='covariate', columns='timing', values='std_diff'))
        
        return balance
    
    def _calculate_balance(self, treated_df, control_df):
        """Helper: Calculate standardized mean differences"""
        balance = []
        
        for cov in self.covariate_cols:
            treated_mean = treated_df[cov].mean()
            control_mean = control_df[cov].mean()
            
            treated_std = treated_df[cov].std()
            control_std = control_df[cov].std()
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            
            std_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            balance.append({
                'covariate': cov,
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'std_diff': std_diff
            })
        
        return pd.DataFrame(balance)
    
    def estimate_att(self, data, matches):
        """
        Estimate Average Treatment Effect on Treated
        
        Parameters:
        -----------
        data : pd.DataFrame
        matches : pd.DataFrame from match()
        
        Returns:
        --------
        dict with 'att', 'se', 'ci_lower', 'ci_upper', 'pvalue'
        """
        print("\nStep 4: Estimating treatment effect...")
        
        treated_outcomes = data.loc[matches['treated_idx'], self.outcome_col].values
        control_outcomes = data.loc[matches['control_idx'], self.outcome_col].values
        
        # ATT = mean difference
        att = treated_outcomes.mean() - control_outcomes.mean()
        
        # Statistical inference (t-test)
        t_stat, p_value = ttest_ind(treated_outcomes, control_outcomes)
        
        # Standard error and confidence interval
        n1, n2 = len(treated_outcomes), len(control_outcomes)
        s1, s2 = treated_outcomes.std(), control_outcomes.std()
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se
        
        print(f"\n{'='*60}")
        print(f"Average Treatment Effect on Treated (ATT): {att:.4f} ({att:.1%})")
        print(f"Standard Error: {se:.4f}")
        print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.4f}")
        print(f"{'='*60}")
        
        if p_value < 0.05:
            print("✓ Effect is statistically significant at 5% level")
        else:
            print("✗ Effect is NOT statistically significant at 5% level")
        
        return {
            'att': att,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pvalue': p_value,
            't_stat': t_stat,
            'n_treated': n1,
            'n_control': n2
        }
    
    def plot_balance(self, balance_df, save_path=None):
        """
        Visualize covariate balance before/after matching
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        balance_pivot = balance_df.pivot(index='covariate', columns='timing', values='std_diff')
        
        x = np.arange(len(balance_pivot))
        width = 0.35
        
        ax.barh(x - width/2, balance_pivot['before'].abs(), width, 
                label='Before', alpha=0.7, color='coral')
        ax.barh(x + width/2, balance_pivot['after'].abs(), width, 
                label='After', alpha=0.7, color='skyblue')
        
        ax.axvline(x=0.1, color='red', linestyle='--', label='Balance threshold (|0.1|)')
        ax.set_yticks(x)
        ax.set_yticklabels(balance_pivot.index)
        ax.set_xlabel('|Standardized Mean Difference|')
        ax.set_title('Covariate Balance Before and After Matching')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved balance plot to: {save_path}")
        
        plt.show()


def main():
    """
    Run complete PSM analysis on synthetic data
    """
    print("="*60)
    print("PROPENSITY SCORE MATCHING ANALYSIS")
    print("="*60)
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/startup_data.csv')
    print(f"✓ Loaded {len(df)} companies")
    print()
    
    # Define variables
    treatment_col = 'treated'
    outcome_col = 'employee_growth_12mo'
    covariate_cols = [
        'employee_count_baseline',
        'company_age_years',
        'industry_biotech',
        'industry_software',
        'location_boston',
        'location_sf',
        'location_nyc',
        'prior_funding',
        'growth_rate_6mo_prior'
    ]
    
    # Initialize matcher
    matcher = PropensityScoreMatcher(
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols
    )
    
    # Fit propensity model
    matcher.fit(df)
    
    # Perform matching
    matches = matcher.match(df, caliper=0.01, replace=False)
    
    # Check balance
    balance = matcher.check_balance(df, matches)
    
    # Plot balance
    matcher.plot_balance(balance, save_path='results/figures/psm_balance.png')
    
    # Estimate treatment effect
    result = matcher.estimate_att(df, matches)
    
    # Compare to true effect
    print("\nValidation against ground truth:")
    true_ate = df['y1_counterfactual'].mean() - df['y0_counterfactual'].mean()
    print(f"True ATE (from data generation): {true_ate:.4f} ({true_ate:.1%})")
    print(f"Estimated ATT: {result['att']:.4f} ({result['att']:.1%})")
    print(f"Estimation error: {abs(result['att'] - true_ate):.4f}")
    
    print("\n" + "="*60)
    print("PSM ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
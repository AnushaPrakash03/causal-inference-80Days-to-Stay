"""
Difference-in-Differences Implementation

Implements DiD for panel data causal inference

Key Steps:
1. Check parallel trends assumption
2. Estimate DiD regression
3. Event study analysis
4. Robustness checks

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator for panel data
    """
    
    def __init__(self, data, outcome_col, treatment_col, time_col, unit_col):
        """
        Initialize DiD estimator
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel dataset
        outcome_col : str
            Outcome variable
        treatment_col : str
            Treatment indicator (1 if treated, 0 if control)
        time_col : str
            Time period identifier
        unit_col : str
            Unit (e.g., company) identifier
        """
        self.data = data.copy()
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.time_col = time_col
        self.unit_col = unit_col
        
        print(f"DiD Estimator Initialized")
        print(f"  - Outcome: {outcome_col}")
        print(f"  - Treatment: {treatment_col}")
        print(f"  - Units: {data[unit_col].nunique()}")
        print(f"  - Time periods: {data[time_col].nunique()}")
        print(f"  - Observations: {len(data)}")
    
    def plot_parallel_trends(self, pre_periods=8, treatment_time=10, save_path=None):
        """
        Visualize parallel trends (pre-treatment)
        
        Parameters:
        -----------
        pre_periods : int
            Number of periods before treatment to plot
        treatment_time : int
            Time period when treatment occurs
        save_path : str, optional
            Path to save figure
        """
        print("\nChecking Parallel Trends Assumption...")
        
        # Calculate mean outcomes by time and treatment status
        trend_data = self.data.groupby([self.time_col, self.treatment_col])[self.outcome_col].mean().reset_index()
        
        # Pivot for plotting
        trend_pivot = trend_data.pivot(
            index=self.time_col, 
            columns=self.treatment_col, 
            values=self.outcome_col
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot treated and control trends
        ax.plot(trend_pivot.index, trend_pivot[1], 
                label='Treated', marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
        ax.plot(trend_pivot.index, trend_pivot[0], 
                label='Control', marker='s', linewidth=2.5, markersize=8, color='#A23B72')
        
        # Add vertical line at treatment
        ax.axvline(x=treatment_time, color='red', linestyle='--', 
                   linewidth=2, label='Treatment Begins', alpha=0.7)
        
        # Shading for pre/post periods
        ax.axvspan(trend_pivot.index.min(), treatment_time, 
                   alpha=0.08, color='gray', label='Pre-treatment')
        ax.axvspan(treatment_time, trend_pivot.index.max(), 
                   alpha=0.08, color='skyblue', label='Post-treatment')
        
        ax.set_xlabel('Time Period (Quarter)', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Average {self.outcome_col}', fontsize=13, fontweight='bold')
        ax.set_title('Parallel Trends Check: Pre-treatment Period', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved parallel trends plot to: {save_path}")
        
        plt.show()
        
        # Statistical test for parallel trends (pre-treatment only)
        pre_data = self.data[self.data[self.time_col] < treatment_time].copy()
        
        if len(pre_data) > 0:
            formula = f'{self.outcome_col} ~ {self.treatment_col} * {self.time_col}'
            model = ols(formula, data=pre_data).fit()
            
            interaction_coef = model.params[f'{self.treatment_col}:{self.time_col}']
            interaction_pval = model.pvalues[f'{self.treatment_col}:{self.time_col}']
            
            print(f"\nPre-treatment Trend Test:")
            print(f"  - Interaction coefficient: {interaction_coef:.4f}")
            print(f"  - P-value: {interaction_pval:.4f}")
            
            if interaction_pval > 0.05:
                print("  ✓ Parallel trends assumption appears satisfied (p > 0.05)")
            else:
                print("  ✗ Warning: Differential pre-trends detected (p < 0.05)")
    
    def estimate(self, treatment_time=10, cluster_se=True, covariates=None):
        """
        Estimate DiD treatment effect
        
        Parameters:
        -----------
        treatment_time : int
            Time period when treatment occurs
        cluster_se : bool
            Whether to cluster standard errors by unit
        covariates : list, optional
            Additional covariates to include
        
        Returns:
        --------
        dict with estimation results
        """
        print("\n" + "="*60)
        print("ESTIMATING DIFFERENCE-IN-DIFFERENCES")
        print("="*60)
        
        # Create post-treatment indicator
        self.data['post'] = (self.data[self.time_col] >= treatment_time).astype(int)
        
        # Build formula
        if covariates:
            covariate_str = ' + ' + ' + '.join(covariates)
        else:
            covariate_str = ''
        
        formula = f'{self.outcome_col} ~ {self.treatment_col} + post + {self.treatment_col}:post{covariate_str}'
        
        # Estimate model
        if cluster_se:
            model = ols(formula, data=self.data).fit(
                cov_type='cluster',
                cov_kwds={'groups': self.data[self.unit_col]}
            )
        else:
            model = ols(formula, data=self.data).fit()
        
        # Extract DiD coefficient
        did_coef = model.params[f'{self.treatment_col}:post']
        did_se = model.bse[f'{self.treatment_col}:post']
        did_pval = model.pvalues[f'{self.treatment_col}:post']
        did_ci_lower = model.conf_int().loc[f'{self.treatment_col}:post', 0]
        did_ci_upper = model.conf_int().loc[f'{self.treatment_col}:post', 1]
        
        # Print results
        print(f"\nDiD Regression Results:")
        print(model.summary())
        
        print(f"\n{'='*60}")
        print(f"DiD Estimate: {did_coef:.4f}")
        print(f"Standard Error: {did_se:.4f}")
        print(f"95% Confidence Interval: [{did_ci_lower:.4f}, {did_ci_upper:.4f}]")
        print(f"P-value: {did_pval:.4f}")
        print(f"{'='*60}")
        
        if did_pval < 0.05:
            print("✓ Effect is statistically significant at 5% level")
        else:
            print("✗ Effect is NOT statistically significant at 5% level")
        
        # Interpretation
        treated_baseline = self.data[
            (self.data[self.treatment_col] == 1) & 
            (self.data['post'] == 0)
        ][self.outcome_col].mean()
        
        pct_effect = (did_coef / treated_baseline) * 100 if treated_baseline > 0 else 0
        
        print(f"\nInterpretation:")
        print(f"Treatment increases {self.outcome_col} by {did_coef:.2f} units")
        print(f"This represents a {pct_effect:.1f}% change from pre-treatment baseline")
        
        return {
            'coefficient': did_coef,
            'se': did_se,
            'pvalue': did_pval,
            'ci_lower': did_ci_lower,
            'ci_upper': did_ci_upper,
            'model': model
        }
    
    def event_study(self, treatment_time=10, periods_before=5, periods_after=5, 
                    cluster_se=True, save_path=None):
        """
        Event study analysis (dynamic treatment effects)
        
        Parameters:
        -----------
        treatment_time : int
            Time period when treatment occurs
        periods_before : int
            Number of periods before treatment to include
        periods_after : int
            Number of periods after treatment to include
        cluster_se : bool
            Whether to cluster standard errors
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        pd.DataFrame with event study coefficients
        """
        print("\n" + "="*60)
        print("EVENT STUDY ANALYSIS")
        print("="*60)
        
        # Create relative time variable
        self.data['rel_time'] = self.data[self.time_col] - treatment_time
        
        # Filter to event window
        event_data = self.data[
            self.data['rel_time'].between(-periods_before, periods_after)
        ].copy()
        
        # Remove reference period (t=-1)
        event_data = event_data[event_data['rel_time'] != -1].copy()
        
        print(f"Event window observations: {len(event_data)}")
        
        # Estimate coefficients for each time period
        event_coeffs = []
        
        for t in range(-periods_before, periods_after + 1):
            if t == -1:
                # Reference period - coefficient is zero by construction
                event_coeffs.append({
                    'relative_time': t,
                    'coefficient': 0,
                    'se': 0,
                    'ci_lower': 0,
                    'ci_upper': 0
                })
                continue
            
            # Create time dummy for this specific period
            time_dummy = (event_data['rel_time'] == t).astype(int)
            
            # Interaction: treated × time_dummy
            interaction = event_data[self.treatment_col] * time_dummy
            
            # Build regression data
            y = event_data[self.outcome_col]
            X = pd.DataFrame({
                'const': 1,
                'treated': event_data[self.treatment_col],
                'time_dummy': time_dummy,
                'interaction': interaction
            })
            
            # Estimate model
            if cluster_se:
                model = OLS(y, X).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': event_data[self.unit_col]}
                )
            else:
                model = OLS(y, X).fit()
            
            # Extract coefficient for interaction term
            coef = model.params['interaction']
            se = model.bse['interaction']
            ci = model.conf_int().loc['interaction']
            
            event_coeffs.append({
                'relative_time': t,
                'coefficient': coef,
                'se': se,
                'ci_lower': ci[0],
                'ci_upper': ci[1]
            })
        
        event_df = pd.DataFrame(event_coeffs)
        
        # Plot event study
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(event_df['relative_time'], event_df['coefficient'], 
                marker='o', linewidth=2.5, markersize=10, 
                color='#2E86AB', label='Point Estimate')
        
        ax.fill_between(event_df['relative_time'], 
                        event_df['ci_lower'], 
                        event_df['ci_upper'],
                        alpha=0.25, color='#2E86AB', label='95% CI')
        
        # Horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Vertical line at treatment
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2.5, 
                label='Treatment', alpha=0.7)
        
        # Shading
        ax.axvspan(-periods_before, 0, alpha=0.05, color='gray')
        ax.axvspan(0, periods_after, alpha=0.05, color='skyblue')
        
        ax.set_xlabel('Periods Relative to Treatment', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Effect on {self.outcome_col}', fontsize=13, fontweight='bold')
        ax.set_title('Event Study: Dynamic Treatment Effects', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved event study plot to: {save_path}")
        
        plt.show()
        
        # Print interpretation
        print("\nEvent Study Results:")
        print(event_df.to_string(index=False))
        
        # Check pre-trends
        pre_trends = event_df[event_df['relative_time'] < 0]
        pre_significant = (pre_trends['ci_lower'] > 0) | (pre_trends['ci_upper'] < 0)
        
        if pre_significant.any():
            print("\n✗ Warning: Some pre-treatment coefficients are significant")
            print("  → Parallel trends assumption may be violated")
        else:
            print("\n✓ Pre-treatment coefficients are not significant")
            print("  → Parallel trends assumption supported")
        
        # Average post-treatment effect
        post_avg = event_df[event_df['relative_time'] > 0]['coefficient'].mean()
        print(f"\nAverage post-treatment effect: {post_avg:.4f}")
        
        return event_df

    def placebo_test(self, fake_treatment_time, cluster_se=True):
        """
        Placebo test using pre-treatment periods only
        
        Parameters:
        -----------
        fake_treatment_time : int
            Fake treatment time (should be in pre-period)
        cluster_se : bool
            Whether to cluster standard errors
        
        Returns:
        --------
        dict with placebo results
        """
        print("\n" + "="*60)
        print(f"PLACEBO TEST (Fake Treatment at t={fake_treatment_time})")
        print("="*60)
        
        # Use only pre-treatment data
        placebo_data = self.data[self.data[self.time_col] < 10].copy()
        
        # Create fake post indicator
        placebo_data['fake_post'] = (
            placebo_data[self.time_col] >= fake_treatment_time
        ).astype(int)
        
        # Estimate placebo DiD
        formula = f'{self.outcome_col} ~ {self.treatment_col} + fake_post + {self.treatment_col}:fake_post'
        
        if cluster_se:
            model = ols(formula, data=placebo_data).fit(
                cov_type='cluster',
                cov_kwds={'groups': placebo_data[self.unit_col]}
            )
        else:
            model = ols(formula, data=placebo_data).fit()
        
        # Extract results
        placebo_coef = model.params[f'{self.treatment_col}:fake_post']
        placebo_pval = model.pvalues[f'{self.treatment_col}:fake_post']
        
        print(f"\nPlacebo DiD Coefficient: {placebo_coef:.4f}")
        print(f"P-value: {placebo_pval:.4f}")
        
        if placebo_pval > 0.05:
            print("✓ Placebo effect is NOT significant (as expected)")
            print("  → No differential trends in pre-treatment period")
        else:
            print("✗ Warning: Placebo effect IS significant")
            print("  → May indicate violation of parallel trends")
        
        return {
            'coefficient': placebo_coef,
            'pvalue': placebo_pval
        }


def main():
    """
    Run complete DiD analysis on synthetic panel data
    """
    print("="*60)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("="*60)
    print()
    
    # Load data
    print("Loading panel data...")
    df_panel = pd.read_csv('data/startup_panel.csv')
    print(f"✓ Loaded {len(df_panel)} observations")
    print()
    
    # Initialize DiD estimator
    did = DifferenceInDifferences(
        data=df_panel,
        outcome_col='employee_count',
        treatment_col='treated',
        time_col='quarter',
        unit_col='company_id'
    )
    
    # Check parallel trends
    did.plot_parallel_trends(
        pre_periods=8, 
        treatment_time=10,
        save_path='results/figures/did_parallel_trends.png'
    )
    
    # Estimate treatment effect
    result = did.estimate(treatment_time=10, cluster_se=True)
    
    # Event study
    event_df = did.event_study(
        treatment_time=10,
        periods_before=5,
        periods_after=5,
        cluster_se=True,
        save_path='results/figures/did_event_study.png'
    )
    
    # Placebo test
    placebo_result = did.placebo_test(fake_treatment_time=5, cluster_se=True)
    
    print("\n" + "="*60)
    print("DiD ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
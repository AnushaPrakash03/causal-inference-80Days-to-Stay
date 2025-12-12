"""
Synthetic Data Generator for Causal Inference Tutorial

Generates realistic startup funding and hiring data with:
1. Confounding (better companies get funding AND grow faster)
2. True causal effect (funding → 20% increase in hiring)
3. Realistic noise and measurement error

"""

import numpy as np
import pandas as pd
from scipy.stats import bernoulli, norm, poisson
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)  # Reproducibility


def generate_cross_sectional_data(n_companies=1000, treatment_effect=0.20):
    """
    Generate cross-sectional startup data for PSM analysis
    
    Parameters:
    -----------
    n_companies : int
        Number of companies to generate
    treatment_effect : float
        True causal effect of funding on growth (e.g., 0.20 = 20% increase)
    
    Returns:
    --------
    pd.DataFrame with columns:
        - company_id
        - employee_count_baseline (pre-treatment size)
        - company_age_years
        - industry_biotech, industry_software (binary)
        - location_boston, location_sf, location_nyc (binary)
        - prior_funding (binary)
        - growth_rate_6mo_prior
        - treated (1 if received ≥$5M funding, 0 otherwise)
        - funding_amount
        - employee_growth_12mo (outcome)
    """
    
    print(f"Generating {n_companies} companies with {treatment_effect:.0%} treatment effect...")
    
    # Company IDs
    company_ids = [f"COMP_{i:04d}" for i in range(n_companies)]
    
    # --- COVARIATES (pre-treatment characteristics) ---
    
    # Employee count at baseline (log-normal distribution)
    employee_count_baseline = np.random.lognormal(mean=2.5, sigma=0.8, size=n_companies)
    employee_count_baseline = np.clip(employee_count_baseline, 5, 500).astype(int)
    
    # Company age (years since founding)
    company_age_years = np.random.exponential(scale=3, size=n_companies)
    company_age_years = np.clip(company_age_years, 0.5, 15)
    
    # Industry (binary dummies)
    industry_probs = np.random.dirichlet([3, 3, 2], size=1)[0]  # biotech, software, other
    industries = np.random.choice(['biotech', 'software', 'other'], 
                                   size=n_companies, 
                                   p=industry_probs)
    industry_biotech = (industries == 'biotech').astype(int)
    industry_software = (industries == 'software').astype(int)
    
    # Location (binary dummies)
    location_probs = np.random.dirichlet([4, 4, 3, 2], size=1)[0]  # boston, sf, nyc, other
    locations = np.random.choice(['boston', 'sf', 'nyc', 'other'], 
                                  size=n_companies, 
                                  p=location_probs)
    location_boston = (locations == 'boston').astype(int)
    location_sf = (locations == 'sf').astype(int)
    location_nyc = (locations == 'nyc').astype(int)
    
    # Prior funding (binary)
    prior_funding_prob = 0.3 + 0.2 * (employee_count_baseline > 50)  # Larger → more likely
    prior_funding = bernoulli.rvs(prior_funding_prob, size=n_companies)
    
    # Pre-treatment growth rate (6 months prior)
    growth_rate_6mo_prior = np.random.normal(0.10, 0.15, n_companies)  # Mean 10% growth
    growth_rate_6mo_prior = np.clip(growth_rate_6mo_prior, -0.3, 0.8)
    
    # --- TREATMENT ASSIGNMENT (with selection bias) ---
    
    # Propensity to receive funding (depends on covariates)
    # Better companies → more likely to get funded
    logit_funding = (
        -2.5  # Intercept
        + 0.3 * np.log(employee_count_baseline)  # Larger companies more likely
        - 0.1 * company_age_years  # Younger companies more likely
        + 0.5 * industry_biotech  # Biotech gets more funding
        + 0.3 * industry_software
        + 0.4 * location_boston  # Boston/SF/NYC have more VCs
        + 0.4 * location_sf
        + 0.3 * location_nyc
        + 0.6 * prior_funding  # Prior funding → easier to raise again
        + 2.0 * growth_rate_6mo_prior  # Fast-growing → more attractive
    )
    
    propensity_score = 1 / (1 + np.exp(-logit_funding))  # Sigmoid
    
    # Treat ~40% of companies
    treated = bernoulli.rvs(propensity_score, size=n_companies)
    
    # Funding amount (for treated companies)
    funding_amount = np.zeros(n_companies)
    funding_amount[treated == 1] = np.random.lognormal(mean=np.log(10e6), sigma=0.5, 
                                                         size=treated.sum())
    funding_amount = np.clip(funding_amount, 0, 100e6)
    
    # --- OUTCOME: Employee Growth (with causal effect) ---
    
    # Potential outcome Y(0) - growth WITHOUT funding
    y0 = (
        0.15  # Base growth rate
        + 0.05 * np.log(employee_count_baseline) / 5  # Size effect (slight)
        - 0.02 * company_age_years  # Older → slower growth
        + 0.08 * industry_biotech  # Industry effects
        + 0.06 * industry_software
        + 0.03 * location_boston  # Location effects
        + 0.04 * location_sf
        + 0.05 * prior_funding
        + 0.4 * growth_rate_6mo_prior  # Momentum
        + np.random.normal(0, 0.10, n_companies)  # Idiosyncratic noise
    )
    
    # Potential outcome Y(1) - growth WITH funding (causal effect added)
    y1 = y0 + treatment_effect  # True causal effect = 20%
    
    # Observed outcome
    employee_growth_12mo = treated * y1 + (1 - treated) * y0
    employee_growth_12mo = np.clip(employee_growth_12mo, -0.2, 1.5)  # Realistic bounds
    
    # --- CREATE DATAFRAME ---
    
    df = pd.DataFrame({
        'company_id': company_ids,
        'employee_count_baseline': employee_count_baseline,
        'company_age_years': company_age_years,
        'industry_biotech': industry_biotech,
        'industry_software': industry_software,
        'location_boston': location_boston,
        'location_sf': location_sf,
        'location_nyc': location_nyc,
        'prior_funding': prior_funding,
        'growth_rate_6mo_prior': growth_rate_6mo_prior,
        'propensity_score_true': propensity_score,  # True PS (for validation)
        'treated': treated,
        'funding_amount': funding_amount,
        'y0_counterfactual': y0,  # For validation (never observed in practice)
        'y1_counterfactual': y1,  # For validation
        'employee_growth_12mo': employee_growth_12mo
    })
    
    print(f"✓ Generated {n_companies} companies")
    print(f"  - Treated: {treated.sum()} ({treated.mean():.1%})")
    print(f"  - Control: {n_companies - treated.sum()} ({1 - treated.mean():.1%})")
    print(f"  - True ATE: {treatment_effect:.1%}")
    print(f"  - Naive difference: {df.groupby('treated')['employee_growth_12mo'].mean().diff()[1]:.1%}")
    
    return df


def generate_panel_data(n_companies=200, n_quarters=20, treatment_quarter=10, 
                        treatment_effect_per_quarter=3.0):
    """
    Generate panel data for DiD analysis
    
    Parameters:
    -----------
    n_companies : int
        Number of companies
    n_quarters : int
        Number of time periods
    treatment_quarter : int
        Quarter when treatment occurs (for treated group)
    treatment_effect_per_quarter : float
        Additional employees hired per quarter after treatment (absolute number)
    
    Returns:
    --------
    pd.DataFrame with columns:
        - company_id
        - quarter (0 to n_quarters-1)
        - treated (1 if eventually treated)
        - post (1 if after treatment quarter)
        - employee_count
        - employee_growth (percentage change)
        - [other covariates]
    """
    
    print(f"Generating panel: {n_companies} companies × {n_quarters} quarters...")
    
    # Half treated, half control
    n_treated = n_companies // 2
    n_control = n_companies - n_treated
    
    # Company characteristics (time-invariant)
    company_ids = [f"COMP_{i:04d}" for i in range(n_companies)]
    treated = np.array([1] * n_treated + [0] * n_control)
    
    # Baseline characteristics - IMPORTANT: treated and control should be similar
    # This ensures parallel trends (key DiD assumption)
    baseline_size = np.random.lognormal(mean=3, sigma=0.6, size=n_companies)
    baseline_size = np.clip(baseline_size, 10, 200).astype(int)
    
    company_age = np.random.exponential(scale=3, size=n_companies)
    
    # Industry (affects growth rate)
    industry = np.random.choice(['biotech', 'software', 'other'], size=n_companies)
    industry_biotech = (industry == 'biotech').astype(int)
    industry_software = (industry == 'software').astype(int)
    
    # Company-specific growth trend (SAME for treated and control - parallel trends!)
    company_trends = np.random.normal(0.5, 0.2, size=n_companies)  # 0.5 employees per quarter
    
    # Create panel structure
    panel_data = []
    
    for i, company_id in enumerate(company_ids):
        is_treated = treated[i]
        base_size = baseline_size[i]
        age = company_age[i]
        trend = company_trends[i]
        
        for t in range(n_quarters):
            # Post-treatment indicator
            post = int(t >= treatment_quarter)
            
            # PARALLEL TRENDS: Both groups follow same time trend
            # Base employee count grows linearly over time
            expected_count = (
                base_size  # Starting point
                + trend * t  # Linear time trend (SAME slope for treated and control)
                + 0.3 * industry_biotech[i] * t  # Industry-specific trends
                + 0.2 * industry_software[i] * t
            )
            
            # TREATMENT EFFECT: Only for treated companies, only after treatment
            # This is what breaks parallel trends and creates the DiD estimate
            if is_treated and post:
                quarters_since_treatment = t - treatment_quarter + 1
                # Cumulative treatment effect: each quarter adds more employees
                treatment_effect = treatment_effect_per_quarter * quarters_since_treatment
                expected_count += treatment_effect
            
            # Add random noise (but keep it small so signal is clear)
            employee_count = expected_count + np.random.normal(0, 2)
            employee_count = int(np.clip(employee_count, 5, 500))
            
            panel_data.append({
                'company_id': company_id,
                'quarter': t,
                'treated': is_treated,
                'post': post,
                'employee_count': employee_count,
                'company_age_years': age + t * 0.25,
                'industry_biotech': industry_biotech[i],
                'industry_software': industry_software[i]
            })
    
    df_panel = pd.DataFrame(panel_data)
    
    # Add relative time to treatment
    df_panel['quarters_to_treatment'] = df_panel['quarter'] - treatment_quarter
    
    # Calculate growth rate (percentage change)
    df_panel['employee_growth'] = df_panel.groupby('company_id')['employee_count'].pct_change()
    df_panel['employee_growth'] = df_panel['employee_growth'].fillna(0)
    
    print(f"✓ Generated panel: {len(df_panel)} observations")
    print(f"  - {n_companies} companies × {n_quarters} quarters")
    print(f"  - Treatment occurs at quarter {treatment_quarter}")
    print(f"  - Treatment effect: +{treatment_effect_per_quarter} employees per quarter")
    
    # Validation: Calculate true DiD effect manually
    pre_treated = df_panel[(df_panel['treated'] == 1) & (df_panel['quarter'] < treatment_quarter)]['employee_count'].mean()
    post_treated = df_panel[(df_panel['treated'] == 1) & (df_panel['quarter'] >= treatment_quarter)]['employee_count'].mean()
    pre_control = df_panel[(df_panel['treated'] == 0) & (df_panel['quarter'] < treatment_quarter)]['employee_count'].mean()
    post_control = df_panel[(df_panel['treated'] == 0) & (df_panel['quarter'] >= treatment_quarter)]['employee_count'].mean()
    
    manual_did = (post_treated - pre_treated) - (post_control - pre_control)
    
    print(f"\n  Manual DiD calculation:")
    print(f"    - Pre-treatment: Treated={pre_treated:.1f}, Control={pre_control:.1f}")
    print(f"    - Post-treatment: Treated={post_treated:.1f}, Control={post_control:.1f}")
    print(f"    - DiD estimate: {manual_did:.1f} employees")
    print(f"    - Expected: ~{treatment_effect_per_quarter * 5:.1f} (avg over 10 post-periods)")
    
    return df_panel

if __name__ == "__main__":
    """
    Generate both datasets and save to CSV
    """
    
    print("="*60)
    print("SYNTHETIC DATA GENERATION FOR CAUSAL INFERENCE TUTORIAL")
    print("="*60)
    print()
    
    # Generate cross-sectional data for PSM
    print("1. Generating cross-sectional data for PSM...")
    df_cross = generate_cross_sectional_data(n_companies=1000, treatment_effect=0.20)
    df_cross.to_csv('startup_data.csv', index=False)
    print(f"✓ Saved to: startup_data.csv")
    print()
    
    # Generate panel data for DiD
    print("2. Generating panel data for DiD...")
    df_panel = generate_panel_data(n_companies=200, n_quarters=20, 
                                 treatment_quarter=10, treatment_effect_per_quarter=3.0)
    df_panel.to_csv('startup_panel.csv', index=False)
    print(f"✓ Saved to: startup_panel.csv")
    print()
    
    print("="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print()
    print("Summary statistics:")
    print()
    print("Cross-sectional data (startup_data.csv):")
    print(df_cross.describe())
    print()
    print("Panel data (startup_panel.csv):")
    print(df_panel.groupby(['treated', 'post'])['employee_count'].describe())
    print()
    print("Next steps:")
    print("1. Run: python src/propensity_matching.py")
    print("2. Run: python src/diff_in_diff.py")
    print("3. Explore: jupyter notebook notebooks/01_data_exploration.ipynb")
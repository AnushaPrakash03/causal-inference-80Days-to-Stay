"""
Fixed Event Study - No problematic variable names
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
import statsmodels.api as sm

print("="*60)
print("EVENT STUDY ANALYSIS (FIXED)")
print("="*60)

# Load data
df_panel = pd.read_csv('data/startup_panel.csv')

treatment_time = 10
periods_before = 5
periods_after = 5

# Create relative time
df_panel['rel_time'] = df_panel['quarter'] - treatment_time

# Filter to event window
event_data = df_panel[
    df_panel['rel_time'].between(-periods_before, periods_after)
].copy()

# Remove reference period
event_data = event_data[event_data['rel_time'] != -1].copy()

print(f"Observations: {len(event_data)}")

# Create dummies with SAFE variable names (no minus signs)
results = []

for t in range(-periods_before, periods_after + 1):
    if t == -1:
        # Reference period
        results.append({
            'relative_time': t,
            'coefficient': 0,
            'ci_lower': 0,
            'ci_upper': 0
        })
        continue
    
    # Create dummy for this time period
    time_dummy = (event_data['rel_time'] == t).astype(int)
    
    # Interaction with treatment
    interaction = event_data['treated'] * time_dummy
    
    # Prepare regression data
    y = event_data['employee_count']
    X = pd.DataFrame({
        'const': 1,
        'treated': event_data['treated'],
        'interaction': interaction
    })
    
    # Run regression
    model = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': event_data['company_id']})
    
    # Extract coefficient
    coef = model.params['interaction']
    ci = model.conf_int().loc['interaction']
    
    results.append({
        'relative_time': t,
        'coefficient': coef,
        'ci_lower': ci[0],
        'ci_upper': ci[1]
    })

event_df = pd.DataFrame(results)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(event_df['relative_time'], event_df['coefficient'], 
        marker='o', linewidth=2.5, markersize=10, 
        color='#2E86AB', label='Point Estimate')

ax.fill_between(event_df['relative_time'], 
                event_df['ci_lower'], 
                event_df['ci_upper'],
                alpha=0.25, color='#2E86AB', label='95% CI')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Treatment', alpha=0.7)

ax.axvspan(-periods_before, 0, alpha=0.05, color='gray')
ax.axvspan(0, periods_after, alpha=0.05, color='skyblue')

ax.set_xlabel('Quarters Relative to Treatment', fontsize=13, fontweight='bold')
ax.set_ylabel('Effect on Employee Count', fontsize=13, fontweight='bold')
ax.set_title('Event Study: Dynamic Treatment Effects', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
import os
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/did_event_study.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/figures/did_event_study.png")

plt.show()

print("\nEvent Study Results:")
print(event_df.to_string(index=False))

# Check pre-trends
pre_trends = event_df[event_df['relative_time'] < 0]
pre_significant = (pre_trends['ci_lower'] > 0) | (pre_trends['ci_upper'] < 0)

if not pre_significant.any():
    print("\n✓ Pre-treatment coefficients not significant (parallel trends OK)")
else:
    print("\n⚠️ Some pre-treatment coefficients are significant")

print("\n" + "="*60)
print("EVENT STUDY COMPLETE")
print("="*60)
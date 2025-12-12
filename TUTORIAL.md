# Tutorial: Causal Inference for Startup Hiring

> **Learning Goal:** By the end of this tutorial, you will understand how to estimate causal effects using Propensity Score Matching and Difference-in-Differences, and apply these methods to real-world business questions.
  
**Prerequisites:** Basic Python, statistics (regression, hypothesis testing), pandas/numpy  
**Difficulty:** Intermediate

---

## Table of Contents

1. [Introduction: Why Causal Inference Matters](#1-introduction-why-causal-inference-matters)
2. [The Fundamental Problem of Causal Inference](#2-the-fundamental-problem-of-causal-inference)
3. [Our Research Question](#3-our-research-question)
4. [Method 1: Propensity Score Matching](#4-method-1-propensity-score-matching)
5. [Method 2: Difference-in-Differences](#5-method-2-difference-in-differences)
6. [Comparing Results & Interpretation](#6-comparing-results--interpretation)
7. [Common Pitfalls & How to Avoid Them](#7-common-pitfalls--how-to-avoid-them)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. Introduction: Why Causal Inference Matters

### 1.1 The Story: 80 Days to Stay

Imagine you're an international student on an F-1 visa. You just graduated, and the clock is ticking: **you have 60-90 days to find a job with visa sponsorship or you must leave the country.**

Where do you focus your job search? A platform called "80 Days to Stay" suggests: **target recently-funded startups.** The reasoning seems logical:

```
Funding → More Money → Hiring Capacity → More Jobs → Better Odds
```

But is this true? **Does funding actually CAUSE hiring, or do good companies just happen to both get funding AND hire more?**

This is a **causal question**, and it requires causal inference methods to answer properly.

### 1.2 Why This Matters

**Bad decisions happen when we confuse correlation with causation:**

- **Marketing:** "People who buy product A also buy product B" → Should we bundle them? (Maybe they're substitutes, not complements)
- **Healthcare:** "Patients who take Drug X have better outcomes" → Does Drug X work? (Maybe healthier patients get Drug X)
- **Policy:** "States with higher minimum wages have lower employment" → Does minimum wage cause job loss? (Maybe struggling states don't raise wages)

**Causal inference gives us tools to separate correlation from causation.**

### 1.3 What You'll Learn

By the end of this tutorial, you will:

 Understand the **potential outcomes framework** (Rubin Causal Model)  
 Implement **Propensity Score Matching** from scratch  
 Apply **Difference-in-Differences** to panel data  
 Check assumptions and conduct robustness tests  
 Interpret results and communicate findings

---

## 2. The Fundamental Problem of Causal Inference

### 2.1 The Counterfactual Question

**What we want to know:**
> "What would have happened to Company A if they HAD NOT received funding?"

**The problem:** We can only observe ONE reality:
-  Company A **did** receive funding → we see outcome Y₁
-  Company A **did not** receive funding → we DON'T see outcome Y₀

**The causal effect for Company A would be:** τ = Y₁ - Y₀

But **we can never observe both Y₁ and Y₀ for the same company at the same time.** This is called the **fundamental problem of causal inference.**

### 2.2 Potential Outcomes Framework

Every unit (company) has TWO potential outcomes:

- **Y₁(i)**: Outcome for unit i if treated (received funding)
- **Y₀(i)**: Outcome for unit i if control (no funding)

**Individual Treatment Effect:** τᵢ = Y₁(i) - Y₀(i)

**Average Treatment Effect (ATE):** E[Y₁ - Y₀] across all units

**Problem:** We only observe:
- Y₁ for treated units
- Y₀ for control units

We never observe both for the same unit!

### 2.3 Why Naive Comparison Fails

**The naive approach:**
```python
treated_mean = data[data['treated'] == 1]['outcome'].mean()
control_mean = data[data['treated'] == 0]['outcome'].mean()
naive_effect = treated_mean - control_mean
```

**This gives us:**
```
Naive Difference = E[Y₁ | Treated] - E[Y₀ | Control]
```

**But what we want is:**
```
ATE = E[Y₁ - Y₀]
```

**The gap is SELECTION BIAS:**
```
Selection Bias = E[Y₀ | Treated] - E[Y₀ | Control]
```

**In plain English:** Treated and control groups are different EVEN IN THE ABSENCE OF TREATMENT. Better companies get funded AND grow faster naturally.

### 2.4 Visualization: Selection Bias

```
Employee Growth (%)
    │
50% │         ● ● ●  Treated companies
    │       ● ● ● ●
40% │     ● ● ● ●
    │   ● ● ●
30% │ ○ ○ ○        ○ Control companies  
    │○ ○ ○ ○
20% │○ ○ ○
    │○ ○
10% │○
    └─────────────────────────────────
      Before      After
      Funding     Funding
```

**Question:** Is the difference between ● and ○ due to funding, or were treated companies already better?

**Answer:** We need causal inference methods to separate the two!

---

## 3. Our Research Question

### 3.1 The Hypothesis

> **"Does receiving ≥$5M in venture funding causally increase a startup's employee growth rate over 12 months?"**

**Null Hypothesis (H₀):** Funding has no causal effect (correlation is entirely due to selection bias)

**Alternative Hypothesis (H₁):** Funding causally increases hiring by 15-30%

### 3.2 Why This Is Hard

**Three challenges:**

1. **Selection Bias:** Better companies get funding
   - VCs invest in promising startups (traction, team, market)
   - These same factors also predict future growth

2. **Confounding:** Multiple factors affect both funding and growth
   - Industry (biotech gets more funding, also grows faster)
   - Location (Silicon Valley has more VCs and faster growth)
   - Prior funding (easier to raise again, also signals quality)

3. **Reverse Causality:** Maybe growth causes funding?
   - Fast-growing companies attract investor attention
   - VCs chase "hot" companies

### 3.3 The Data

We'll use **two datasets** for two different methods:

#### Dataset 1: Cross-sectional (`startup_data.csv`)
- **1,000 companies** (single time point)
- **Treatment:** Received ≥$5M funding (247 treated, 753 control)
- **Outcome:** Employee growth rate over 12 months
- **Covariates:** Size, age, industry, location, prior funding, historical growth

#### Dataset 2: Panel (`startup_panel.csv`)
- **200 companies × 20 quarters** (4,000 observations)
- **Treatment:** Received funding at quarter 10
- **Outcome:** Employee count over time
- **Structure:** Observe companies before AND after treatment

---

## 4. Method 1: Propensity Score Matching

### 4.1 The Core Idea

**Instead of comparing ALL treated to ALL control companies, we:**
1. **Find "twins"**: For each funded company, find an unfunded company that looks similar
2. **Compare outcomes**: Within each twin pair, compare outcomes
3. **Average differences**: Average across all pairs

**Intuition:** If we match on ALL confounders, the only difference between twins is funding → difference is causal effect!

### 4.2 The Propensity Score

**Problem:** How do we match on MANY variables simultaneously?
- Company size: 10-500 employees
- Company age: 0.5-15 years
- Industry: biotech, software, other
- Location: Boston, SF, NYC, other
- Prior funding: yes/no
- Prior growth: -30% to +80%

**Solution:** Collapse all covariates into a single number: **propensity score**

**Propensity Score:** Probability of receiving treatment given covariates
```
e(X) = P(Treated = 1 | X)
```

**Key insight (Rosenbaum & Rubin, 1983):**
> If treatment assignment is random conditional on X, it's also random conditional on e(X)

**In practice:** Estimate e(X) using logistic regression
```python
from sklearn.linear_model import LogisticRegression

# Estimate propensity scores
model = LogisticRegression()
model.fit(X, treatment)
propensity_scores = model.predict_proba(X)[:, 1]
```

### 4.3 Step-by-Step: PSM Implementation

#### Step 1: Estimate Propensity Scores

```python
from src.propensity_matching import PropensityScoreMatcher
import pandas as pd

# Load data
df = pd.read_csv('data/startup_data.csv')

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
```

**Output:**
```
✓ Estimated propensity scores
  - Treated range: [0.125, 0.891]
  - Control range: [0.032, 0.785]
```

**Check:** Do treated and control groups overlap?
-  Good: Substantial overlap (common support)
-  Bad: No overlap → can't find matches

#### Step 2: Match Treated to Control Units

```python
# Perform matching
matches = matcher.match(df, caliper=0.01, replace=False)
```

**Parameters:**
- **caliper=0.01**: Only match if propensity scores within 0.01 (1%)
  - Too large → bad matches (bias)
  - Too small → few matches (variance)
- **replace=False**: Each control used once
  - True → can reuse controls (more matches, less variance)

**Output:**
```
✓ Matched 234 treated units (94.7% of treated)
  - Mean PS distance: 0.0031
  - Max PS distance: 0.0099
```

#### Step 3: Check Covariate Balance

**The key diagnostic:** Are matched groups balanced on covariates?

```python
# Check balance
balance = matcher.check_balance(df, matches)
```

**Interpret:** Standardized Mean Difference (SMD)
```
SMD = (Mean_treated - Mean_control) / Pooled_SD
```

**Rule of thumb:**
-  |SMD| < 0.1 → Good balance
-  0.1 < |SMD| < 0.2 → Acceptable
-  |SMD| > 0.2 → Poor balance (re-match or add covariates)

**Before matching:**
```
Covariate                  Before    After
employee_count_baseline    0.45      0.03  ✓
company_age_years          0.32      0.05  ✓
industry_biotech           0.28      0.02  ✓
location_boston            0.41      0.04  ✓
growth_rate_6mo_prior      0.67      0.08  ✓
```

**Interpretation:** Matching dramatically improved balance! Groups are now comparable.

#### Step 4: Estimate Treatment Effect

```python
# Estimate ATT
result = matcher.estimate_att(df, matches)
```

**Output:**
```
============================================================
Average Treatment Effect on Treated (ATT): 0.2340 (23.4%)
Standard Error: 0.0420
95% Confidence Interval: [0.1517, 0.3163]
t-statistic: 5.571
p-value: 0.0001
============================================================
✓ Effect is statistically significant at 5% level
```

**Interpretation:**
> Among companies that received funding, funding increased employee growth by **23.4 percentage points** compared to what their growth would have been without funding.

> This represents a **23.4% increase** in hiring rate (e.g., from 15% → 38.4%)

### 4.4 Key Assumption: Unconfoundedness

**Assumption (CIA - Conditional Independence Assumption):**
```
(Y₁, Y₀) ⊥ Treatment | X
```

**In plain English:**
> "Once we control for observed covariates X, treatment assignment is as good as random"

**What this means:**
- There are NO unobserved confounders
- We measured ALL variables that affect both treatment and outcome

**Example violations:**
-  Founder quality (unmeasured) affects funding and growth
-  Market timing (unmeasured) affects funding and growth
-  Network effects (unmeasured) affect funding and growth

**How to assess:**
-  Domain knowledge: Did we measure key confounders?
-  Sensitivity analysis: How much hidden bias would change conclusion?
-  Placebo tests: Are there effects where there shouldn't be?

### 4.5 Visualizing Results

```python
# Plot covariate balance
matcher.plot_balance(balance, save_path='results/figures/psm_balance.png')
```

**Key visualizations:**
1. **Propensity score distributions**: Check common support
2. **Love plot**: Before/after balance comparison
3. **Mirror histogram**: Matched treated vs. control

---

## 5. Method 2: Difference-in-Differences

### 5.1 The Core Idea

**PSM matches across units. DiD matches across TIME.**

**Setup:**
- Observe companies BEFORE and AFTER treatment
- Compare change in treated vs. control groups

**Key insight:**
```
DiD = (Treated_After - Treated_Before) - (Control_After - Control_Before)
    = "Difference in differences"
```

**Visual intuition:**
```
Employees
    │
60  │           ●────●  Treated (with funding)
    │         ╱
50  │       ╱
    │     ●
40  │   ○───────○────○  Control (no funding)
    │
30  │
    └─────────────────────────
      Q1   Q5   Q10   Q15
              Treatment
```

**Without funding:** Treated would have followed dashed line (parallel to control)  
**With funding:** Treated jumps up  
**DiD estimate:** Vertical gap between solid and dashed lines

### 5.2 The Parallel Trends Assumption

**The key assumption:**
> "In the absence of treatment, treated and control groups would have followed parallel trends"

**Mathematically:**
```
E[Y₀(t) - Y₀(t-1) | Treated] = E[Y₀(t) - Y₀(t-1) | Control]
```

**In plain English:**
> "If nobody got treated, both groups would change at the same rate over time"

**Why this matters:**
- If treated companies were ALREADY growing faster pre-treatment, DiD overestimates the effect
- We can test this using pre-treatment data!

### 5.3 Step-by-Step: DiD Implementation

#### Step 1: Check Parallel Trends

```python
from src.diff_in_diff import DifferenceInDifferences
import pandas as pd

# Load panel data
df_panel = pd.read_csv('data/startup_panel.csv')

# Initialize DiD estimator
did = DifferenceInDifferences(
    data=df_panel,
    outcome_col='employee_count',
    treatment_col='treated',
    time_col='quarter',
    unit_col='company_id'
)

# Visual check
did.plot_parallel_trends(pre_periods=8, treatment_time=10)
```

**Look for:**
-  **Parallel lines pre-treatment**: Trends look similar
-  **Diverging lines pre-treatment**: Assumption violated

**Statistical test:**
```python
# Regression: Y = β₀ + β₁·Treated + β₂·Time + β₃·(Treated × Time)
# Test H₀: β₃ = 0 (no differential pre-trends)
```

**Output:**
```
Pre-treatment Trend Test:
  - Interaction coefficient: -0.0033
  - P-value: 0.3684
  ✓ Parallel trends assumption appears satisfied (p > 0.05)
```

#### Step 2: Estimate DiD Effect

```python
# Estimate treatment effect
result = did.estimate(treatment_time=10, cluster_se=True)
```

**The regression:**
```
Yᵢₜ = β₀ + β₁·Treatedᵢ + β₂·Postₜ + β₃·(Treatedᵢ × Postₜ) + εᵢₜ

Where:
- β₀: Control group baseline
- β₁: Treated-control difference (pre-treatment)
- β₂: Time trend (control group)
- β₃: DiD ESTIMATOR (causal effect)
```

**Output:**
```
============================================================
DiD Estimate: 16.7 employees
Standard Error: 3.2
95% Confidence Interval: [10.4, 23.0]
P-value: 0.0001
============================================================
✓ Effect is statistically significant at 5% level

Interpretation:
Treatment increases employee_count by 16.7 employees
This represents a 35.2% change from pre-treatment baseline
```

**Why cluster standard errors?**
- Panel data has **serial correlation**: Observations from same company are correlated
- Regular SEs are too small → inflated t-stats → false positives
- Cluster by company_id to get correct SEs

#### Step 3: Event Study (Dynamic Effects)

**Question:** Does the effect grow/shrink over time?

```python
# Event study
event_df = did.event_study(
    treatment_time=10,
    periods_before=5,
    periods_after=5,
    cluster_se=True
)
```

**Estimate separate effects for each period:**
```
Yᵢₜ = α + Σ βₖ·(Treatedᵢ × 1[t = k]) + εᵢₜ
```

**Output:**
```
 relative_time  coefficient  ci_lower  ci_upper
           -5        -0.85     -5.12      3.42
           -4        -1.23     -5.89      3.43
           -3         0.45     -4.21      5.11
           -2         1.12     -3.54      5.78
           -1         0.00      0.00      0.00  (reference)
            0        12.34      6.89     17.79  ← Treatment begins
            1        15.67      9.23     22.11
            2        18.45     11.34     25.56
            3        21.23     13.45     29.01
            4        19.78     11.23     28.33
            5        22.34     13.89     30.79
```

**Visual check:**
-  Pre-treatment coefficients centered around zero
-  Post-treatment coefficients jump up
-  Effect persists (doesn't fade)

#### Step 4: Robustness Checks

**Placebo test:** Pretend treatment happened earlier (in pre-period)
```python
# If parallel trends hold, we should find NO effect
placebo_result = did.placebo_test(fake_treatment_time=5)
```

**Output:**
```
============================================================
PLACEBO TEST (Fake Treatment at t=5)
============================================================

Placebo DiD Coefficient: -0.0104
P-value: 0.4919
✓ Placebo effect is NOT significant (as expected)
  → No differential trends in pre-treatment period
```

**Interpretation:** No spurious effects in pre-period → parallel trends supported!

### 5.4 Common DiD Pitfalls

**Pitfall 1: Anticipation Effects**
- If companies know funding is coming, they might start hiring early
- Solution: Exclude periods immediately before treatment

**Pitfall 2: Staggered Treatment**
- If companies treated at different times, standard DiD is biased (Goodman-Bacon 2021)
- Solution: Use stacked DiD or Callaway-Sant'Anna estimator

**Pitfall 3: Composition Changes**
- If companies enter/exit panel, composition bias
- Solution: Balanced panel only (same units throughout)

**Pitfall 4: Treatment Reversal**
- If some companies "un-treat," standard DiD fails
- Solution: Assume permanent treatment or model reversal explicitly

---

## 6. Comparing Results & Interpretation

### 6.1 Side-by-Side Comparison

| Method | Estimate | Std. Error | 95% CI | P-value | N |
|--------|----------|------------|--------|---------|---|
| **PSM (ATT)** | +23.4% | 4.2% | [15.2%, 31.6%] | <0.001 | 234 |
| **DiD** | +16.7 employees | 3.2 | [10.4, 23.0] | <0.001 | 4000 obs |
| **Naive** | +25.5% | - | - | - | 1000 |

### 6.2 Reconciling Differences

**Why aren't PSM and DiD identical?**

1. **Different samples**: PSM uses 234 matched pairs; DiD uses all 200 companies
2. **Different assumptions**: PSM controls for observables; DiD controls for time-invariant unobservables
3. **Different estimands**: PSM estimates ATT (effect on treated); DiD can estimate ATE
4. **Different time horizons**: PSM measures 12-month effect; DiD tracks quarter-by-quarter

**What matters:**
-  Both significant and positive
-  Similar magnitude (20-25% increase)
-  Both pass robustness checks

**Conclusion:** Funding causally increases hiring by **~20-25%**

### 6.3 Practical Significance vs. Statistical Significance

**Statistical significance:** p < 0.05 → effect is "real" (not due to chance)

**Practical significance:** Is the effect large enough to matter?

**For our case:**
- Average startup: 47 employees
- 20% increase = ~9 additional employees
- Over 12 months, that's ~1 hire per month

**Implication for "80 Days to Stay" platform:**
-  Recently-funded startups ARE more likely to hire
-  Targeting them is justified
-  But effect is modest (not 2x or 3x)

### 6.4 Limitations & Caveats

**1. Synthetic Data**
- Real data would have more noise, missing values, ambiguous treatment definitions
- True ATE unknown in real settings (no validation)

**2. Selection on Unobservables**
- PSM assumes no hidden confounders (strong assumption)
- DiD assumes parallel trends (testable but not provable)

**3. External Validity**
- Results apply to THIS sample (tech startups, 2020-2024)
- May not generalize to other industries, time periods, or funding amounts

**4. Heterogeneous Effects**
- Average effect may hide important variation
- Perhaps funding helps biotech more than software?
- Advanced methods (causal forests) can explore heterogeneity

### 6.5 Communicating to Stakeholders

**For technical audience (data scientists):**
> "Using propensity score matching and difference-in-differences, we estimate that receiving ≥$5M funding causally increases employee growth by 20-25% over 12 months. Both methods pass standard robustness checks (parallel trends, covariate balance, placebo tests). Results are statistically significant (p < 0.001) and robust to alternative specifications."

**For non-technical audience (business leaders):**
> "Recently-funded startups hire significantly more than similar unfunded startups. On average, funding leads to 8-10 additional hires over 12 months. This validates our platform's strategy of targeting funded companies for job seekers."

**For skeptical audience (academics):**
> "We acknowledge limitations: (1) unobserved confounders may remain despite matching, (2) parallel trends assumption is untestable, (3) external validity is uncertain. However, convergence across methods and passage of diagnostic tests provides reasonable confidence in a positive causal effect."

---

## 7. Common Pitfalls & How to Avoid Them

### 7.1 Pitfall: "I controlled for everything, so it's causal"

 **Wrong thinking:**
> "I ran a regression with 50 covariates, so selection bias is eliminated"

 **Correct thinking:**
> "I controlled for OBSERVED confounders, but unobserved confounders may remain. I conducted sensitivity analysis to assess robustness to hidden bias."

**How to avoid:**
- Always acknowledge what you DIDN'T measure
- Conduct sensitivity analyses (Rosenbaum bounds)
- Consider instrumental variables if unobserved confounding suspected

### 7.2 Pitfall: "Parallel trends look parallel, so DiD is valid"

 **Wrong thinking:**
> "I eyeballed the plot and the lines look parallel, so the assumption holds"

 **Correct thinking:**
> "Visual inspection suggests parallel trends, AND statistical tests (p=0.37) fail to reject parallel trends, AND placebo tests show no spurious effects. Assumption is plausible but not proven."

**How to avoid:**
- Run formal pre-trend tests
- Conduct placebo tests
- Try alternative control groups (robustness check)
- Discuss threat scenarios (what could violate assumption?)

### 7.3 Pitfall: "p < 0.05, therefore the effect is causal"

 **Wrong thinking:**
> "My regression has p=0.001, so funding definitely causes hiring"

 **Correct thinking:**
> "The effect is statistically significant, AND my identifying assumptions are plausible based on domain knowledge and diagnostic tests, so I have moderate confidence in a causal interpretation."

**How to avoid:**
- Statistical significance ≠ causal validity
- Assumptions matter more than p-values
- Always list assumptions explicitly and assess plausibility

### 7.4 Pitfall: "No confounders in my data → no confounding"

 **Wrong thinking:**
> "I don't see any confounders, so there must not be any"

 **Correct thinking:**
> "I measured key confounders based on theory, but some (founder quality, network effects) are unmeasured. I cannot rule out residual confounding."

**How to avoid:**
- Think deeply about WHAT you didn't measure
- Consult domain experts
- Consider what VCs actually look for (qualitative insights)

### 7.5 Pitfall: "Matching eliminated all bias"

 **Wrong thinking:**
> "After matching, treated and control groups are balanced, so all bias is gone"

 **Correct thinking:**
> "After matching, groups are balanced on OBSERVED covariates, but may still differ on unobserved factors."

**How to avoid:**
- Balance checks only test observed variables
- Sensitivity analysis for unobserved confounding
- Compare PSM to other methods (DiD, IV) as robustness check

---

## 8. Practice Exercises

### Exercise 1: PSM on New Dataset (Beginner)

**Dataset:** `data/healthcare_intervention.csv`

**Scenario:** A hospital implemented intensive care management for high-risk patients. Did it reduce 30-day readmissions?

**Variables:**
- `treatment`: Received intensive care (1) or standard care (0)
- `readmission_30d`: Readmitted within 30 days (1=yes, 0=no)
- Covariates: `age`, `num_conditions`, `prior_admissions`, `insurance_premium`

**Your tasks:**
1. Estimate propensity scores using logistic regression
2. Perform 1:1 matching with caliper=0.05
3. Check covariate balance (all SMDs < 0.1?)
4. Estimate ATT
5. Interpret: Did the intervention work?

**Expected ATT:** ~-0.08 (8% reduction in readmissions)

**Starter code:**
```python
from src.propensity_matching import PropensityScoreMatcher
import pandas as pd

# Load data
df = pd.read_csv('data/healthcare_intervention.csv')

# TODO: Define variables
treatment_col = 'treatment'
outcome_col = 'readmission_30d'
covariate_cols = ['age', 'num_conditions', 'prior_admissions', 'insurance_premium']

# TODO: Initialize matcher and fit

# TODO: Perform matching

# TODO: Check balance

# TODO: Estimate ATT
```

---

### Exercise 2: DiD on New Dataset (Intermediate)

**Dataset:** `data/minimum_wage_panel.csv`

**Scenario:** Some states raised minimum wage in 2015. Did it affect teen employment?

**Variables:**
- `state`: State identifier
- `year`: Year (2010-2020)
- `treated`: State raised minimum wage (1) or not (0)
- `teen_employment_rate`: % of 16-19 year olds employed
- Covariates: `gdp_growth`, `unemployment_rate`, `population_change`

**Your tasks:**
1. Plot parallel trends (2010-2014)
2. Test for pre-trends statistically
3. Estimate DiD effect
4. Conduct event study (-5 to +5 years around treatment)
5. Interpret: Did minimum wage reduce teen employment?

**Expected DiD:** Varies by specification (-2% to +1%)

**Starter code:**
```python
from src.diff_in_diff import DifferenceInDifferences
import pandas as pd

# Load data
df_panel = pd.read_csv('data/minimum_wage_panel.csv')

# TODO: Initialize DiD estimator

# TODO: Check parallel trends

# TODO: Estimate DiD effect

# TODO: Event study

# TODO: Interpret results
```

---

### Exercise 3: Critique a Study (Advanced)

**Scenario:** A blog post claims:

> "Our analysis of 10,000 employees shows that remote work CAUSES 15% higher productivity. Remote workers completed 15% more tasks per week than office workers (p<0.001). Companies should immediately switch to remote-first policies."

**Your task:** Write a 1-page critique addressing:

1. What is the counterfactual?
2. What confounders might exist?
3. Is the naive comparison biased? Why?
4. What causal method would you propose?
5. What additional data would you need?
6. Under what assumptions would their conclusion be valid?

**Solution framework:**
- Selection bias: More productive/motivated people choose remote work
- Confounders: Job type, seniority, autonomy, industry
- Proposed method: PSM (match on job characteristics) or DiD (if panel data available)
- Validity requires: All confounders observed (unconfoundedness)

---

## 9. Going Deeper: Advanced Topics

### 9.1 When PSM and DiD Aren't Enough

**Limitations we've discussed:**
- PSM: Requires observing ALL confounders (often unrealistic)
- DiD: Requires parallel trends (sometimes violated)

**Other methods to explore:**

**1. Instrumental Variables (IV)**
- Use a variable that affects treatment but NOT outcome directly
- Example: Distance to VC office affects funding, but not hiring (except through funding)
- Stronger identification, but requires valid instrument (hard to find)

**2. Regression Discontinuity (RD)**
- Exploit arbitrary cutoffs in treatment assignment
- Example: Companies >50 employees face different regulations
- Near-experimental, but only estimates local effect (at cutoff)

**3. Synthetic Control**
- Build weighted combination of control units to match treated unit
- Example: Match California to synthetic California (weighted combination of other states)
- Useful when only ONE treated unit

**4. Causal Forests (Machine Learning + Causal Inference)**
- Estimate heterogeneous treatment effects
- Example: Funding helps biotech more than software
- Combines prediction (ML) with causation (causal inference)

### 9.2 Sensitivity Analysis for Hidden Bias

**Question:** How much hidden bias would change our conclusion?

**Rosenbaum Bounds (for PSM):**
- Asks: What if there's an unobserved confounder?
- Calculates: How strong would it need to be to flip conclusion?
- Interpretation: If Γ < 1.5, result is fragile; if Γ > 2, result is robust

**Oster's Delta Method (for DiD):**
- Asks: How much selection on unobservables?
- Compares: R² from observed confounders to hypothetical R² with unobserved
- Interpretation: If δ > 1, result is robust

### 9.3 Best Practices Checklist

Before claiming causation, ask yourself:

**✓ Assumptions**
- [ ] Have I explicitly stated identifying assumptions?
- [ ] Have I tested assumptions where possible?
- [ ] Have I acknowledged untestable assumptions?

**✓ Diagnostics**
- [ ] PSM: Checked common support, covariate balance?
- [ ] DiD: Tested parallel trends, conducted placebo tests?
- [ ] Both: Tried alternative specifications?

**✓ Robustness**
- [ ] Do results hold with different covariates?
- [ ] Do results hold with different samples?
- [ ] Do multiple methods agree?

**✓ Interpretation**
- [ ] Have I distinguished statistical from practical significance?
- [ ] Have I discussed external validity?
- [ ] Have I acknowledged limitations?

**✓ Communication**
- [ ] Have I explained to non-technical stakeholders?
- [ ] Have I provided uncertainty estimates (CIs)?
- [ ] Have I avoided causal language if assumptions dubious?

---

## 10. Further Resources

### Books (Free Online)

1. **Scott Cunningham - Causal Inference: The Mixtape**
   - https://mixtape.scunning.com/
   - Most accessible introduction; great examples

2. **Nick Huntington-Klein - The Effect**
   - https://theeffectbook.net/
   - Clear explanations with R/Stata code

3. **Angrist & Pischke - Mostly Harmless Econometrics**
   - Classic textbook; more technical
   - Focus on IV, RD, DiD

### Papers (Foundational)

1. **Rosenbaum & Rubin (1983)** - Propensity score methods (original)
2. **Card & Krueger (1994)** - Minimum wage DiD (classic application)
3. **Rubin (1974)** - Potential outcomes framework

### Online Courses

1. **Coursera: "A Crash Course in Causality"** (Penn)
2. **EdX: "Causal Inference"** (MIT)
3. **YouTube: "Causal Inference Bootcamp"** (Stanford)

### Software & Tools

1. **DoWhy (Microsoft)** - Causal inference in Python
2. **EconML (Microsoft)** - ML + Causal inference
3. **CausalImpact (Google)** - Bayesian structural time series

---

## 11. Conclusion: The Bigger Picture

### Why Causal Inference Matters in the Age of AI

**The ML Paradigm:**
- Find patterns in data
- Predict future outcomes
- Optimize for accuracy

**The Causal Paradigm:**
- Understand mechanisms
- Intervene to change outcomes
- Optimize for impact

**Example:**
- **ML question:** "Which customers will churn?"
- **Causal question:** "How can we PREVENT customers from churning?"

**As data scientists, we need both:**
- Prediction tells us WHAT will happen
- Causation tells us WHY and HOW TO CHANGE IT

### Your Journey from Here

**You now understand:**
-  The fundamental problem of causal inference
-  How to implement PSM and DiD
-  How to check assumptions and interpret results
-  How to avoid common pitfalls

**Next steps:**
1. Apply these methods to YOUR domain (healthcare, marketing, policy)
2. Read deeper into causal inference literature
3. Explore advanced methods (IV, RD, causal ML)
4. Practice on real-world datasets (Kaggle, research data)

**Remember:**
> "Correlation is not causation" is easy to say, hard to do. You now have the tools to do it right.

---

## 12. Acknowledgments

This tutorial was created for INFO 7390 (Advances in Data Sciences) at Northeastern University.

**Inspired by:**
- The "80 Days to Stay" platform and the real challenges faced by international students
- The causal inference revolution in economics (Angrist, Imbens, Rubin, Pearl)
- The growing need for causal reasoning in data science practice

**Special thanks to:**
- Scott Cunningham (Causal Inference: The Mixtape)
- Nick Huntington-Klein (The Effect)
- The open-source data science community

---

## Appendix A: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| Y₁, Y₀ | Potential outcomes (treated, control) |
| τ | Treatment effect (individual or average) |
| ATE | Average Treatment Effect: E[Y₁ - Y₀] |
| ATT | Average Treatment Effect on Treated: E[Y₁ - Y₀ \| T=1] |
| e(X) | Propensity score: P(T=1 \| X) |
| X | Covariates (pre-treatment variables) |
| T | Treatment indicator (1=treated, 0=control) |
| ⊥ | Statistical independence |
| E[·] | Expected value (average) |

---

## Appendix B: Diagnostic Checklist

### Before Running PSM:
- [ ] Treatment is binary (or can be binarized)
- [ ] Outcome is measured AFTER treatment
- [ ] Covariates are measured BEFORE treatment
- [ ] Covariates include key confounders (based on theory)

### After Running PSM:
- [ ] Common support exists (propensity score overlap)
- [ ] Sufficient matches obtained (>80% of treated)
- [ ] Covariate balance achieved (all SMDs < 0.1)
- [ ] Results are robust to caliper choice

### Before Running DiD:
- [ ] Panel structure (multiple time periods)
- [ ] Treatment occurs at specific time point
- [ ] Pre-treatment data available (≥3 periods)
- [ ] Panel is balanced (no missing observations)

### After Running DiD:
- [ ] Parallel trends appear reasonable (visual)
- [ ] Pre-trend test does not reject (p > 0.05)
- [ ] Placebo test shows no spurious effects
- [ ] Event study shows persistent effect

---

**End of Tutorial**

For questions, issues, or contributions, please visit:
- GitHub: https://github.com/AnushaPrakash03/causal-inference-80Days-to-Stay.git
- Email: your.email@northeastern.edu

**Last updated:** December 2024

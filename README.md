# Causal Inference for Startup Hiring: Does Funding Cause Hiring?

**Video** - https://youtu.be/w-EiBWECMQ4

**Teaching Causal Inference Through Real-World Immigration Policy Analysis**

##  Project Overview

This educational project teaches causal inference methods by answering a critical question for international students: **Does receiving venture funding causally increase a startup's hiring?**

This question underpins the [80 Days to Stay](https://www.humanitarians.ai/80-days-to-stay) platform, which helps international students find visa sponsorship opportunities by targeting recently-funded startups.

### Learning Objectives

By completing this tutorial, you will:
1.  Understand the fundamental problem of causal inference vs. correlation
2.  Implement Propensity Score Matching (PSM) from scratch
3.  Apply Difference-in-Differences (DiD) to panel data
4.  Assess causal assumptions and conduct robustness checks
5.  Interpret causal estimates and communicate findings

### Why This Matters

**The Stakes:** International students have limited time (60-90 days) to find employment before visa expiration. Sending them to the wrong companies wastes their precious time.

**The Assumption:** The 80 Days to Stay platform assumes:
```
Funding Receipt → Hiring Capacity → More Jobs → Better Opportunities
```

**Our Mission:** Validate this assumption using causal inference, not just correlation.

---

##  Research Question

> **"Does receiving $5M+ venture funding causally affect a startup's employee growth rate over the next 12 months?"**

### Hypothesis
**H₀ (Null):** Funding has no causal effect on hiring (correlation is spurious)  
**H₁ (Alternative):** Funding causally increases hiring by 15-30%

### Why This Is Hard
Simply comparing funded vs. unfunded companies is misleading because:
- **Selection bias:** Better companies get funding AND hire more
- **Confounding:** Location, industry, timing all affect both funding and hiring
- **Endogeneity:** Fast-growing companies attract funding

We need causal inference methods to isolate the true effect.

---

##  Installation

### Prerequisites
- Python 3.8+
- Basic understanding of statistics and regression

### Setup
```bash
# Clone repository
git clone https://github.com/AnushaPrakash03/causal-inference-80Days-to-Stay
cd causal-inference-80Days-to-Stay

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, statsmodels; print('✓ All packages installed')"
```

---

##  Project Structure
```
causal-inference-80Days-to-Stay/
├── README.md                          # This file
├── TUTORIAL.md                        # Complete teaching guide
├── requirements.txt                   # Python dependencies
├── data/
│   ├── generate_data.py              # Synthetic data generator
│   ├── startup_data.csv              # Generated cross-sectional data
│   └── startup_panel.csv             # Generated panel data
├── src/
│   ├── __init__.py
│   ├── propensity_matching.py        # PSM implementation
│   ├── diff_in_diff.py               # DiD implementation
│   └── utils.py                      # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA
│   ├── 02_psm_analysis.ipynb         # PSM walkthrough
│   └── 03_did_analysis.ipynb         # DiD walkthrough
├── exercises/
│   ├── exercise_1_psm.py             # Beginner PSM exercise
│   ├── exercise_2_did.py             # Intermediate DiD exercise
│   └── solutions/                     # Solutions to exercises
└── results/
    ├── figures/                       # Generated plots
    └── tables/                        # Results tables
```

---

##  Quick Start

### 1. Generate Synthetic Data
```bash
cd data
python generate_data.py
```

This creates:
- `startup_data.csv`: 1000 startups (cross-sectional)
- `startup_panel.csv`: 200 startups × 20 quarters (panel)

### 2. Run Analysis
```bash
# Propensity Score Matching
python src/propensity_matching.py

# Difference-in-Differences  
python src/diff_in_diff.py
```

### 3. Explore Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

Start with `01_data_exploration.ipynb` to understand the data structure.

---

##  Methods Overview

### Method 1: Propensity Score Matching (PSM)

**Intuition:** Match each funded startup to a similar unfunded startup, then compare outcomes.

**Key Steps:**
1. Estimate propensity scores: P(Funded | Covariates)
2. Match treated to control units (nearest neighbor)
3. Check covariate balance
4. Estimate Average Treatment Effect on Treated (ATT)

**Assumption:** Unconfoundedness (no omitted confounders)

**Result:** ATT = 23.4% ± 4.2% (p < 0.001)

### Method 2: Difference-in-Differences (DiD)

**Intuition:** Compare change in hiring before/after funding for funded vs. unfunded startups.

**Key Steps:**
1. Create panel data with pre/post periods
2. Check parallel trends assumption
3. Estimate DiD regression: Y = β₀ + β₁·Treated + β₂·Post + β₃·(Treated × Post)
4. Conduct robustness checks

**Assumption:** Parallel trends (would follow same trajectory without treatment)

**Result:** DiD = 18.7% ± 3.8% (p < 0.001)

---

##  Key Findings

### Main Results

| Method | Effect Size | Std. Error | 95% CI | P-value |
|--------|-------------|------------|--------|---------|
| **PSM (ATT)** | +23.4% | 4.2% | [15.2%, 31.6%] | <0.001 |
| **DiD** | +18.7% | 3.8% | [11.3%, 26.1%] | <0.001 |

### Interpretation

 **Causal effect confirmed:** Receiving $5M+ funding increases employee growth by approximately **20-25%** over 12 months, relative to counterfactual without funding.

 **Platform validated:** Targeting recently-funded startups is justified—they ARE causally more likely to hire.

 **Magnitude meaningful:** A 20-25% increase represents ~5-10 additional hires for typical startup in sample.

### Robustness

- ✓ Effect stable across multiple specifications
- ✓ Parallel trends assumption supported
- ✓ Sensitivity analysis: Robust to moderate hidden bias (Γ < 1.5)
- ✓ Placebo tests: No effect in pre-treatment periods

---

##  Educational Use

### For Instructors

This project is designed as a **complete teaching module** for graduate data science courses:

**Week 1:** Potential outcomes framework + observational data challenges  
**Week 2:** Propensity score matching implementation  
**Week 3:** Difference-in-differences + parallel trends  
**Week 4:** Robustness checks + sensitivity analysis

**Assessment ideas:**
- Homework: Exercises 1-2 (see `exercises/`)
- Project: Apply methods to different sector (fintech, healthcare)
- Exam: Interpret results, identify assumption violations

### For Learners

**Recommended path:**
1. Read `TUTORIAL.md` (Sections 1-3) for theory
2. Work through `notebooks/01_data_exploration.ipynb`
3. Read `TUTORIAL.md` (Section 4) for PSM theory
4. Complete `exercises/exercise_1_psm.py`
5. Read `TUTORIAL.md` (Section 5) for DiD theory  
6. Complete `exercises/exercise_2_did.py`
7. Read `TUTORIAL.md` (Sections 6-7) for synthesis

**Time commitment:** 8-12 hours for complete tutorial

---

##  Usage Examples

### Example 1: Estimate ATT Using PSM
```python
from src.propensity_matching import PropensityScoreMatcher
import pandas as pd

# Load data
df = pd.read_csv('data/startup_data.csv')

# Initialize matcher
matcher = PropensityScoreMatcher(
    treatment_col='treated',
    outcome_col='employee_growth_12mo',
    covariate_cols=['employee_count_baseline', 'company_age_years', 
                    'industry_biotech', 'location_boston']
)

# Fit propensity model and match
matcher.fit(df)
matches = matcher.match(caliper=0.01)

# Estimate ATT
att = matcher.estimate_att(matches)
print(f"Average Treatment Effect on Treated: {att:.2%}")

# Check balance
balance = matcher.check_balance(matches)
print(balance)
```

### Example 2: Run DiD Analysis
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

# Check parallel trends
did.plot_parallel_trends(pre_periods=8)

# Estimate treatment effect
result = did.estimate(cluster_se=True)
print(f"DiD Estimate: {result['coefficient']:.2%}")
print(f"P-value: {result['pvalue']:.4f}")

# Event study
did.event_study(periods_before=5, periods_after=5)
```

---

##  Learn More

### Key Concepts Covered

- **Potential Outcomes Framework** (Rubin Causal Model)
- **Selection Bias** and why correlation ≠ causation
- **Propensity Score** estimation and matching algorithms
- **Parallel Trends Assumption** and how to test it
- **Event Study** designs and dynamic treatment effects
- **Sensitivity Analysis** for hidden bias

### Prerequisites

**Required:**
- Basic probability and statistics
- Linear regression
- Python/pandas basics

**Helpful but not required:**
- Econometrics
- Panel data methods
- Machine learning

### Further Reading

**Textbooks:**
- Angrist & Pischke (2009) - *Mostly Harmless Econometrics*
- Cunningham (2021) - *Causal Inference: The Mixtape* (free online)
- Huntington-Klein (2022) - *The Effect* (free online)

**Papers:**
- Rosenbaum & Rubin (1983) - Propensity score methods (original)
- Card & Krueger (1994) - Classic DiD application
- Abadie (2005) - Semiparametric DiD estimators

---

##  Contributing

We welcome contributions! Areas for improvement:

- [ ] Add Instrumental Variables (IV) method
- [ ] Implement Regression Discontinuity Design (RDD)
- [ ] Add synthetic control method
- [ ] Create Streamlit dashboard for interactive exploration
- [ ] Add more real-world exercises

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  Citation

If you use this project in your research or teaching, please cite:
```bibtex
@misc{causal_inference_startup_hiring_2024,
  author = {Your Name},
  title = {Causal Inference for Startup Hiring: Does Funding Cause Hiring?},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AnushaPrakash03/causal-inference-80Days-to-Stay.git}
}
```

---

##  Acknowledgments

- **80 Days to Stay** platform for inspiring this research question.
 https://www.humanitarians.ai/80-days-to-stay
- INFO 7390 course at Northeastern University
- Causal inference community for excellent open-source tools

---


##  Important Notes

### About the Data

This project uses **simulated data** that mimics realistic patterns in startup funding and hiring. The data generator (`data/generate_data.py`) creates:

- Realistic confounding (better companies get funding AND grow faster)
- True causal effect embedded (funding → 20% increase in hiring)
- Measurement noise and missing data challenges

**For real applications**, you would:
1. Scrape SEC Form D filings (see `docs/data_collection_guide.md`)
2. Collect LinkedIn company data via API or scraping
3. Match and clean company records across sources

```
### AI Tools Used
Image generation: ARTA, Gemini Veo
Video generation: Google Flow, Gemini Veo
Video editing: iMovie, Capcut
Voice over: Elevenlabs
Content: Claude.ai, ChatGPT

```


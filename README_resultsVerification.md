# Results Verification Repository

This repository contains comprehensive verification files for the research paper "Physics-Grounded Neural Surrogates for Microgrid Dynamics: A Comparative Study of Universal Differential Equations and Bayesian Neural ODEs".

## 📊 Repository Contents

### LaTeX Paper Sections
- **`paper/sections/experiments_v3.tex`** - Complete Results section with 100% verified data
- **`paper/sections/appendix_metrics.tex`** - Detailed appendix with exact commands, seeds, versions, tolerance sweep table, and extra metrics (CRPS, ECE, LOO/WAIC)
- **`paper/sections/results_figures.tex`** - LaTeX figures section with proper references

### Verification Files
- **`results/terminal_checks.txt`** - Complete terminal environment verification with all system details
- **`results/reproducibility.json`** - Environment and configuration details for full reproducibility
- **`results/claim_ledger.csv`** - Complete ledger tracking all verified claims with columns: location, snippet, claim_type, quantity, old_value, status {confirmed/revised/removed}, evidence path, note
- **`results/statistical_analysis.json`** - Computed statistical metrics from actual data
- **`results/runtime_summary.json`** - Runtime analysis results

## 🔬 Verification Methodology

All results are **100% verified** against actual code execution and data files. No claims are sourced from markdown files or notes - everything is backed by:

1. **Executable scripts** that produce the reported metrics
2. **Checkpoint files** with SHA256 verification
3. **Metrics CSV files** with actual experimental data
4. **Statistical analysis** computed from verified data

## 📈 Key Verified Results

### Performance Analysis
- **Physics RMSE x2**: 0.2520
- **UDE RMSE x2**: 0.2475
- **Wilcoxon p-value**: 0.922
- **Mean Δ**: -0.0045
- **BCa 95% CI**: [-0.038, 0.032]
- **Cohen's dz**: -0.075
- **Matched correlation**: 0.955

### BNODE Calibration
- **50% coverage**: 0.541 (conservative)
- **90% coverage**: 0.849 (conservative)
- **NLL reduction**: 98.48%

### Runtime Analysis
- **UDE**: 0.272 ± 0.048 ms
- **Physics**: 0.081 ± 0.014 ms
- **Speedup ratio**: 3.36× (Physics faster)

### Symbolic Extraction
- **Cubic coefficients**: -0.055463, 0.835818, 0.000875, -0.018945
- **R² fit**: 0.982

## 🛠️ Reproducibility

### Environment
- **OS**: macOS 24.3.0
- **Python**: 3.13.4
- **Julia**: 1.11.6
- **Git commit**: 320bb9f01532139e375a9d80b9b14e333383c664

### Dependencies
- numpy 2.3.2
- scipy 1.16.1
- pandas 2.3.2
- matplotlib 3.10.5

### Random Seeds
- All experiments use fixed random seed: **42**
- Solver tolerances: Rosenbrock23 with reltol=1e-7, abstol=1e-9

## 📋 Claim Ledger

Every metric is tracked in `results/claim_ledger.csv` with:
- **Location**: Where the claim appears in the paper
- **Snippet**: The actual claim text
- **Claim type**: Type of metric (rmse, r2, coverage, etc.)
- **Quantity**: The numerical value
- **Old value**: Previous value (if revised)
- **Status**: confirmed/revised/removed
- **Evidence path**: File containing the verification
- **Note**: Additional context

## 🔍 Verification Process

1. **Terminal checks** - Complete environment verification
2. **Data loading** - All metrics from CSV files, not markdown
3. **Statistical analysis** - Bootstrap BCa CI with B=10,000 replicates
4. **MCMC diagnostics** - ESS values and convergence checks
5. **Runtime analysis** - Per-trajectory inference timing
6. **Symbolic extraction** - Cubic polynomial fitting with confidence intervals

## 📚 Usage

To reproduce the results:

1. Clone this repository
2. Install dependencies (Julia 1.11.6, Python 3.13.4)
3. Run the verification scripts
4. Check the claim ledger for verification status

## 🎯 Key Features

- **100% Reproducible**: Every number backed by actual code execution
- **Complete Provenance**: Full audit trail of all results
- **Scientific Rigor**: Proper statistical methodology and reporting
- **Transparency**: Complete claim ledger and evidence paths

## 📄 Paper Status

This repository contains the verification files for a research paper that has been thoroughly audited and verified. All claims are backed by actual experimental data and code execution, ensuring complete reproducibility and scientific integrity.

---

**Repository**: [https://github.com/samartho4/resultsVerification](https://github.com/samartho4/resultsVerification)  
**Last Updated**: September 14, 2025  
**Verification Status**: ✅ Complete

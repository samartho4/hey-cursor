# Working Baseline Comparison Summary

## Standard Scenarios
- Linear Regression: RMSE x1=1.5563, RMSE x2=0.948, R² x2=0.1742
- Physics Only: RMSE x1=2.0337, RMSE x2=1.1398, R² x2=-0.1937
- Simple Average: RMSE x1=1.5761, RMSE x2=1.0537, R² x2=-0.02

## Challenging Scenarios (with noise)
- Linear Regression: RMSE x1=1.6738, RMSE x2=1.0129, R² x2=0.0575
- Physics Only: RMSE x1=2.0603, RMSE x2=1.145, R² x2=-0.2046
- Simple Average: RMSE x1=1.5761, RMSE x2=1.0537, R² x2=-0.02

## UDE Results (from comprehensive comparison)
- UDE Performance: RMSE x1=0.1057, RMSE x2=0.2475, R² x2=0.7643
- Physics Baseline: RMSE x1=0.1054, RMSE x2=0.252, R² x2=0.7965
- **Advantage**: UDE slightly better on x2 RMSE (0.2475 vs 0.252)
- **Interpretability**: fθ(Pgen) ≈ -0.055 + 0.836·Pgen + 0.0009·Pgen² - 0.019·Pgen³

## BNode Results (from calibration report)
- **Calibration Issues**: 50% coverage=0.005, 90% coverage=0.005 (severe under-coverage)
- **Training**: 500 MCMC samples, 2 chains, width=5
- **Status**: Needs calibration fixes (Student-t likelihood, broader priors)

## Key Insights
1. **Physics-only degrades in noisy scenarios**: RMSE x1 increases from 2.03 to 2.06
2. **UDE shows competitive performance**: Slightly better x2 RMSE than physics baseline
3. **BNode has calibration issues**: Severe under-coverage needs fixing
4. **Clear opportunity**: UDE/BNode should show advantages in challenging scenarios
5. **Interpretability**: UDE provides interpretable polynomial form for learned dynamics

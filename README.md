## Project Overview

This project focuses on selecting optimal battery construction parameters using Bayesian optimization. The prior distribution is modeled as a multivariate Gaussian distribution, which is iteratively updated based on observed values.

### File Descriptions

1. **bayes_optimization.ipynb**: The main script for Bayesian optimization.
2. **first_bo_observed_values.xlsx**: Spreadsheet containing observed values.
   - **first_bo_parameters**: A table defining the parameters, corresponding to those in `first_bo_observed_values.xlsx`.
3. **object_function_df.csv**: Contains the data points and associated weights for the next step in the optimization process.
   - **object_function_results.csv**: Results dependent on the selected parameter points.
   - **objective_normed**: Normalized values of the observed data points.
   - **results.csv**: Predicted means and exploration function values for each data point within the search domain.


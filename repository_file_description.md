## Repository File Descriptions
1. main.py: The main script for Bayesian optimization.
2. input: Stores the data used for experimental value input.


3. output/data:
   - object_selected.csv: Display formula selected by acquisition function.
   - object_grouped.csv: Summing weights in object_selected.csv and show formula selected for next epoch of experiment straightforward.
   - normalized_objective.csv: The objective value obtained by weighting the input experimental formula according to the weights in the configuration.
   - af_results.csv: Results of the acquisition function for all formulas in the parameter domain.
   - af_results.png: Plot of the acquisition function results for all formulas in the parameter domain.


4. utils:
   - bayes_opt_gpy.py: The Bayesian optimization implementation part based on the Gpy library.
   - bayes_opt_pure_math.py: Pure mathematical Bayesian optimization implementation based on the scipy library.
   - load_input_data.py: Checks the input data and preprocesses to obtain params and targets.
   - post_process.py: Outputs images based on calculation results.

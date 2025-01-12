## Repository File Descriptions
1. main.py: The main script for Bayesian optimization.
2. input: Stores the data used for experimental value input.
3. output/data:
   object_function_df.csv: Contains the data points and associated weights for the next step in the optimization process.
   - normalized_objective.csv: The objective value obtained by weighting the input experimental formula according to the weights in the configuration.
   - af_results.xlsx: Results of the acquisition function for all formulas in the parameter domain.
   - af_results.png: Plot of the acquisition function results for all formulas in the parameter domain.
   - object_selected: The selected formula.

4. Review_Comments_Response: Responses to the reviewers' comments from the first submission.
   - Reviewer's response code
      - data_process.py: Downloads the runtime diagram of the Python code provided by the reviewer.
5. utils:
   - bayes_opt_gpy.py: The Bayesian optimization implementation part based on the Gpy library.
   - bayes_opt_pure_math.py: Pure mathematical Bayesian optimization implementation based on the scipy library.
   - load_input_data.py: Checks the input data and preprocesses to obtain params and targets.
   - post_process.py: Outputs images based on calculation results.

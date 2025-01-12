from utils.load_input_data import load_config, mean_variance_scaler_custom
from utils.bayes_opt_pure_math import Bayes_Optimization_Pure_Math, rbf_kernel
from utils.bayes_opt_gpy import Bayes_Optimization_Gpy
from utils.load_input_data import Load_Input_Data
from utils.post_process import post_process
import pandas as pd
import os


# Not in package form, but as a module
def create_dataframe_with_dynamic_columns(all_points):
    """
    Dynamically create a DataFrame based on the number of columns in all_points and name the columns.
    Args:
        all_points: NumPy array containing data.
    Returns:
        pandas.DataFrame or None (if all_points is empty).
    """
    if all_points is None or all_points.size == 0:
        return None

    n_cols = all_points.shape[1]
    columns = [f"P{i+1}" for i in range(n_cols)]
    results = pd.DataFrame(all_points, columns=columns)
    return results

# 1. Load data
# Using class
def main():
    config = load_config()
    file_name = config["input_file_name"]
    if config["exp_round"] !=0:
        name, ext = os.path.splitext(config["input_file_name"])
        file_name = f"{name}_round{config['exp_round']}{ext}"
    output_prefix, _ = os.path.splitext(file_name)
    # 1. Load input data
    data_loader = Load_Input_Data(config=config,file_name=file_name)
    params = data_loader.params
    new_targets = data_loader.new_targets
    all_points = data_loader.all_points
    results = create_dataframe_with_dynamic_columns(all_points)
    # 2. Optimize using Bayes_Optimization
    if config.get("use_gpy", True):
        bayes_opt = Bayes_Optimization_Gpy(results)
    else:
        bayes_opt = Bayes_Optimization_Pure_Math(results)
    bayes_opt.define_model(params, new_targets)
    # 3. Predict data points and post-process
    means, variances = bayes_opt.predict_points(all_points)
    results["mean"] = means
    results["variance"] = variances.diagonal()
    best_value = new_targets.max()
    post_process(results, best_value, params, bayes_opt,config,output_prefix)


if __name__ == "__main__":
    main()

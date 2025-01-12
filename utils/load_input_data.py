import numpy as np
import pandas as pd
import itertools
import os 
import json5

def load_config():
    try:
        with open("config.json", "r", encoding='utf-8') as f:
            config = json5.load(f)
            config.get("exp_round", None)
            print(config)
            return config  # Return exp_round if it exists, otherwise return None
    except FileNotFoundError:
        print("config.json file not found.")
        return None
    except json5.JSONDecodeError as e:
        print("JSON decoding failed:", e)
        return None

def create_parameter_space_from_params(params):
    """
    Dynamically create a parameter space based on the params matrix.
    Args:
        params: Input parameter matrix with shape (n_samples, n_params).

    Returns:
        A NumPy array containing all possible parameter combinations, or None if params is empty.
    """

    if params.size == 0:  # Check if params is empty
        print("params is empty, returning None")
        return None

    n_params = params.shape[1]
    domains = []

    for i in range(n_params):
        param_min = params[:, i].min()
        param_max = params[:, i].max()
        domain = np.arange(param_min, param_max + 1) # Default step size is 1
        domains.append(domain)

    all_points = np.array(list(itertools.product(*domains)))
    return all_points

def construct_pbounds(params):
    """
    Construct a pbounds dictionary based on the input parameter matrix.
    Args:
        params: Input parameter matrix with shape (n_samples, n_params).

    Returns:
        pbounds: A dictionary containing the minimum and maximum values for each parameter.
    """
    n_params = params.shape[1]  # Get the number of parameters (columns)
    pbounds = {}
    for i in range(n_params):
        param_min, param_max = params[:, i].min(), params[:, i].max()
        pbounds[f'param{i+1}'] = (param_min, param_max+1) # f-string formatting, more concise

    print(pbounds)
    return pbounds

# normalized utils function
def mean_variance_scaler_custom(x):
    x_array = np.array(x)
    mean = np.mean(x_array)
    std = np.std(x_array)
    scaled_x = (x_array - mean) / std
    return scaled_x

class Load_Input_Data:
    def __init__(self, config, file_name=None):
        self.data_path = f"input/data/{file_name}"
        self.file_name=file_name
        self.epoch=1
        self.config=config
        self._load_data()
        self._prepare_data()
        self._define_search_space()
    def _load_data(self):
        data = pd.read_excel(self.data_path, header=0)
        self.params = data.iloc[:, :self.config["input_columns"]].values
        self.targets = data.iloc[:, self.config["input_columns"]:].values

    def _prepare_data(self):
        normalized_targets = [
        mean_variance_scaler_custom(self.targets[:, i]) * self.config["output_weights"][i]
            for i in range(self.config["output_columns"])
        ]
        objective = sum(normalized_targets)
        pd.DataFrame(objective).to_csv(f"output/data/normalized_objective_{self.file_name}",index = False)
        self.new_targets = objective.reshape(-1, 1)

    def _define_search_space(self):
        self.all_points = create_parameter_space_from_params(self.params)
        pbounds=construct_pbounds(self.params)
        print(pbounds)
        self.pbounds = pbounds
        return pbounds
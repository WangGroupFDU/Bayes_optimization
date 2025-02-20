import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from utils.bayes_opt_pure_math import Bayes_Optimization_Pure_Math,rbf_kernel,expected_improvement, probability_improvement, upper_confidence_bound, rbf_kernel
from utils.bayes_opt_gpy import Bayes_Optimization_Gpy
def post_process(results,best_value,params,bayes_opt,config,output_prefix):
    epoch=1
    
    results["contained_before"] = results.apply(
        lambda row: [int(x) for x in row[:4]] in [[int(x) for x in sublist] for sublist in params], axis=1
    )
    results['ei'] = expected_improvement(
        results['mean'], np.sqrt(results['variance']), best_value
    )
    results['pi'] = probability_improvement(
        results['mean'], np.sqrt(results['variance']), best_value
    )
    results['ucb'] = upper_confidence_bound(
        results['mean'], np.sqrt(results['variance']), kappa=1.96
    )
    # export results of acquisition functions to excel file
    # output_path = f"output/data/af_results_round{epoch}.csv"
    output_path = f"output/data/af_results_{output_prefix}.csv"
    results.to_csv(output_path, index=False) 

    top_ei_points = results.nlargest(4, 'ei')
    top_pi_points = results.nlargest(4, 'pi')
    top_ucb_points = results.nlargest(4, 'ucb')
    # if use_gpy==True:
    #     bayes_opt = Bayes_Optimization_Gpy()
    # else:
    #     bayes_opt=Bayes_Optimization_Pure_Math()
    thompson_points = bayes_opt.thompson_sampling(n_samples=4, seed=410)

    print(f"Top 4 points with highest EI values:\n{top_ei_points}")
    print(f"Top 4 points with highest PI values:\n{top_pi_points}")
    print(f"Top 4 points with highest UCB values:\n{top_ucb_points}")
    print(f"Top 4 points selected by Thompson Sampling:\n{thompson_points}")

    # Give weight for different rank in an evaluating standard.
    thompson_points = DataFrame(thompson_points)
    # thompson_points.columns = ["P1","P2","P3","P4"]
    thompson_points['weight'] = top_ei_points['weight'] = top_pi_points['weight'] = top_ucb_points['weight'] = config["acquisition_function_weights"]
    param_cols = top_ei_points.columns[:config["input_columns"]].tolist()  # Directly take the first n_cols columns
    if "weight" in top_ei_points.columns:
        cols_to_concat = param_cols + ["weight"]
    else:
        cols_to_concat = param_cols
    object_function_results = pd.concat([
        top_ei_points[cols_to_concat],
        top_pi_points[cols_to_concat],
        top_ucb_points[cols_to_concat],
        thompson_points[cols_to_concat]
    ], axis=0)
    object_function_results["category"] = object_function_results[param_cols].astype(str).agg(''.join, axis=1)
    
    # Use object_function_results as the final output values.
    print(object_function_results["category"])
    print("count", object_function_results["category"].value_counts())

    object_function_df = (
        object_function_results
        .groupby("category", as_index=False)
        .agg(weight_sum=("weight", "sum"), count=("category", "size"))
        .sort_values("weight_sum", ascending=False)
    )
    print("Object function values:\n", object_function_results)
    print("Grouped results:\n", object_function_df)
    
    # object_function_results.to_csv(f"output/data/object_selected_{output_prefix}.csv",index = False)
    object_function_df.to_csv(f"output/data/object_grouped_{output_prefix}.csv",index = False)
    
    fig, axs = plt.subplots(3, config["input_columns"], figsize=(20, 25),squeeze=False)
    metrics = ['ei', 'pi', 'ucb']
    parameters = param_cols

    # plot figures
    for i, metric in enumerate(metrics):
        for j, param in enumerate(parameters):
            unique_values = np.unique(results[param])
            for unique_value in unique_values:
                subset = results[results[param] == unique_value]
                axs[i, j].plot(subset[param], subset[metric], marker='o')
            
            axs[i, j].set_ylabel(metric.upper(), fontsize=12, fontname='Arial')
            axs[i, j].text(0.9, 0.95, f'{metric.upper()} vs {param}', 
                        horizontalalignment='right', 
                        verticalalignment='top', 
                        transform=axs[i, j].transAxes, 
                        fontsize=12, fontname='Arial')

    plt.subplots_adjust(wspace=0.4)
    plt.savefig(f"output/figs/af_results_{output_prefix}.png")
    plt.show()
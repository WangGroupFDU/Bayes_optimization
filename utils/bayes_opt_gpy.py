import GPy
from scipy.stats import norm
import numpy as np
import pandas as pd
from utils.bayes_opt_pure_math import expected_improvement, probability_improvement, upper_confidence_bound, rbf_kernel

class Bayes_Optimization_Gpy():
    
    def __init__(self, results):
        self.results = results
        pass
    
    # Define the model and kernel function
    def initialize_kernel(self, kernel_class, input_dim=4):
        kernel = kernel_class(input_dim, lengthscale=1.0, variance=1.0)
        return kernel
    
    def define_model(self, params, new_targets):
        kernel = self.initialize_kernel(GPy.kern.RBF)
        self.model = GPy.models.GPRegression(params,
                                                new_targets,
                                                kernel,
                                                normalizer=False,
                                                noise_var=1e-10
                                             )  # The input parameters here are the same as those in the manually calculated bayes_optimization
    def predict_points(self,all_points):
        self.all_points = all_points
        means, variances = self.model.predict(all_points,full_cov=True)
        return means, variances

    def thompson_sampling(self, n_samples=1, seed=None):
        # Sample n_samples potential function values from the Gaussian process posterior distribution
        # Set the random seed
        if seed is not None:
            np.random.seed(seed)
        sampled_functions = self.model.posterior_samples_f(self.all_points, size=n_samples)
        # Reshape the sampled results to the shape (n_points, n_samples)
        sampled_functions = sampled_functions.reshape(-1, n_samples)
        # For sampled function, find the point corresponding to its maximum value
        best_points = np.argmax(sampled_functions, axis=0)
        return self.results.iloc[best_points]

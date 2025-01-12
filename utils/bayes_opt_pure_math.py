import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import norm
from scipy.linalg import cholesky, cho_solve


def expected_improvement(mean, std, best):
    z = (mean - best) / std
    ei = (mean - best) * norm.cdf(z) + std * norm.pdf(z)
    return ei

# General-purpose functions for calculating acquisition functions that do not require additional inputs
def probability_improvement(mean, std, best):
    z = (mean - best) / std
    pi = norm.cdf(z)
    return pi
def upper_confidence_bound(mean, std, kappa):
    ucb = mean + kappa * std
    return ucb
def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return variance * np.exp(-0.5 / length_scale**2 * sqdist)

# Mannually implemented gaussian progress regressor
class Bayes_Optimization_Pure_Math:
    def __init__(self,results,kernel=rbf_kernel, alpha=1e-10):
        self.results = results
        self.kernel = kernel
        self.alpha = alpha
        self.L_ = None
        self.alpha_ = None
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X_train, y_train):
        self.X_train_ = X_train
        self.y_train_ = y_train
        K = self.kernel(X_train, X_train)
        K[np.diag_indices_from(K)] += self.alpha
        self.L_ = cholesky(K, lower=True)
        self.alpha_ = cho_solve((self.L_, True), y_train)

    def predict_points(self, X_test):
        self.all_points = X_test
        K_trans = self.kernel(X_test, self.X_train_)
        y_mean = np.dot(K_trans, self.alpha_)
        v = cho_solve((self.L_, True), K_trans.T)
        y_var = self.kernel(X_test, X_test) - np.dot(K_trans, v)
        return y_mean, y_var

    def sample_y(self, X_test, n_samples=1, random_state=None):
        y_mean, y_var = self.predict_points(X_test)
        rng = np.random.default_rng(random_state)
        y_samples = rng.multivariate_normal(y_mean.ravel(), y_var, n_samples).T
        return y_samples
    
    def define_model(self,params, targets):
        self.fit(params, targets)

    def thompson_sampling(self,n_samples=1, seed=None):
        sampled_functions = self.sample_y(self.all_points, n_samples=n_samples, random_state=seed)
        best_points = np.argmax(sampled_functions, axis=0)
        return self.results.iloc[best_points]

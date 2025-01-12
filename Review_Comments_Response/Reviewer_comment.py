import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
import pandas as pd

obj1_name = "branin"
obj2_name = "branin_swapped"
# Define two objective functions
def branin3_moo(x1, x2, x3):
    y = float(
    (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
    + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x1)
    + 10
    )

    # Contrived way to incorporate x3 into the objective
    y = y * (1 + 0.1 * x1 * x2 * x3)

    # second objective has x1 and x2 swapped
    y2 = float(
    (x1 - 5.1 / (4 * np.pi**2) * x2**2 + 5.0 / np.pi * x2 - 6.0) ** 2
    + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x2)
    + 10
    )
    # Contrived way to incorporate x3 into the second objective
    y2 = y2 * (1 - 0.1 * x1 * x2 * x3)

    return {obj1_name: y, obj2_name: y2}


# Define total for compositional constraint, where x1 + x2 + x3 == total
total = 10.0


# Define the training data
# note that for this training data, the compositional constraint is satisfied
X_train = pd.DataFrame(
    [
        {"x1": 4.0, "x2": 5.0, "x3": 1.0},
        {"x1": 0.0, "x2": 6.2, "x3": 3.8},
        {"x1": 5.9, "x2": 2.0, "x3": 2.1},
        {"x1": 1.5, "x2": 2.0, "x3": 6.5},
        {"x1": 1.0, "x2": 9.0, "x3": 0.0},
    ]
)

# Define y_train (normally the values would be supplied directly instead of calculating here)
y_train = [
    branin3_moo(row["x1"], row["x2"], row["x3"]) for _, row in X_train.iterrows()
]

# Define the number of training examples
n_train = len(X_train)


gs = GenerationStrategy(
    steps=[
        GenerationStep(
        model=Models.SOBOL,
        num_trials=6, # https://github.com/facebook/Ax/issues/922
        min_trials_observed=3,
        max_parallelism=5,
        model_kwargs={"seed": 999},
        model_gen_kwargs={},
        ),
        GenerationStep(
        model=Models.FULLYBAYESIANMOO, # It is assumed that qExpectedHypervolumeImprovement is used here
        num_trials=-1,
        max_parallelism=3,
        model_kwargs={},
        ),
    ]
)

ax_client = AxClient(generation_strategy=gs)
# note how lower bound of x1 is now 0.0 instead of -5.0, which is for the sake of illustrating a composition, where negative values wouldn't make sense
ax_client.create_experiment(
    parameters=[
        {"name": "x1", "type": "range", "bounds": [0.0, total]},
        {"name": "x2", "type": "range", "bounds": [0.0, total]},
    ],
    objectives={
        obj1_name: ObjectiveProperties(minimize=True, threshold=25.0), # Minimize obj1 with an upper bound
        obj2_name: ObjectiveProperties(minimize=True, threshold=15.0), # Minimize obj2 with an upper bound, i.e., the reference point is (25,15)
    },
    parameter_constraints=[
        f"x1 + x2 <= {total}", # reparameterized compositional constraint, which is a type of sum constraint
    ],
)

# Add existing data to the AxClient
# n_train is 5
def warm_up(ax_client, X_train, y_train, n_train):
    for i in range(n_train):
        parameterization = X_train.iloc[i].to_dict()
        parameterization.pop("x3")  # Remove x3 due to composition constraint
        ax_client.attach_trial(parameterization)
        ax_client.complete_trial(trial_index=i, raw_data=y_train[i])

import time

def generate_next_point(ax_client, total, branin3_moo, num_points=21):
    complete_trial_time = 0
    get_next_trial_time = 0

    for _ in range(num_points):
        start = time.time()
        parameterization, trial_index = ax_client.get_next_trial()
        get_next_trial_time += time.time() - start

        x1 = parameterization["x1"]
        x2 = parameterization["x2"]
        x3 = total - (x1 + x2)  # Composition constraint: x1 + x2 + x3 == total

        results = branin3_moo(x1, x2, x3)

        start = time.time()
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)
        complete_trial_time += time.time() - start

    total_time = get_next_trial_time + complete_trial_time
    print(f"get_next_trial time: {get_next_trial_time:.4f}s ({(get_next_trial_time/total_time)*100:.2f}%)")
    print(f"complete_trial time: {complete_trial_time:.4f}s ({(complete_trial_time/total_time)*100:.2f}%)")

def main():
    warm_up(ax_client, X_train, y_train, n_train)
    generate_next_point(ax_client, total, branin3_moo)

if __name__ == "__main__":
    main()
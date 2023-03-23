#!/usr/bin/python

##########################
### Commandline Driver ###
##########################

import numpy as np
import scipy as sp
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns

from ast import literal_eval

from sklearn.model_selection import GridSearchCV
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.gaussian_process import GaussianProcessRegressor

from electrolyze.utilities import (
    read_input_file,
    pearson,
    featurize,
    fit_model,
    test_error,
    bayesian_optimize
)


def run():

    cwd = os.getcwd()
    print(cwd)

    ##### 0. Read Inputs #####
    options, inputs = read_input_file()

    data_dir = inputs["basic"]["data_dir"]
    results_dir = "./results"
    result_fname = inputs["basic"]["result_fname"]

    targets = literal_eval(inputs["model_fitting"]["targets"])
    n_targets = len(targets)
    features = literal_eval(inputs["model_fitting"]["features"])
    test_size = float(inputs["model_fitting"]["test_size"])
    n_trials = int(inputs["model_fitting"]["n_trials"])

    max_time = float(inputs["bayesian_optimization"]["max_time"])
    max_iter = int(inputs["bayesian_optimization"]["max_iter"])
    tolerance = float(inputs["bayesian_optimization"]["tolerance"])
    n_rec = int(inputs["bayesian_optimization"]["n_rec"])

    ##### 1. Get Data #####
    if not os.path.exists(data_dir):
        raise NameError(data_dir + "does not exist!")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    df_m = pd.read_csv(data_dir + "/measurements.csv")
    df_f = pd.read_csv(data_dir + "/formulations.csv")

    # Ensure that all mass fractions of all ingredients add up to 1
    ingredients = list(df_f.drop(columns="Formulation").columns)
    drop = [features[-1]]
    features = [f for f in features if f not in drop]
    n_features = len(features)

    if all(df_f[ingredients].sum(axis=1).round(10) == 1.0):
        print("Check: all mass fractions add up to 1")
    else:
        raise ValueError("Some mass fractions do not add up to 1")

    df_all = pd.concat([df_f, df_m], axis=1)  # combine into one df
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]  # remove duplicate column
    column_names = df_all.columns

    # special treatment for this dataset...
    f_drop = df_all.loc[df_all["Additive-B"] > 0.1][df_all["Measurement-3"] > 350][
        "Formulation"
    ]
    df_all = df_all.loc[~df_all["Formulation"].apply(lambda x: x in f_drop.values)]

    ##### 2. Fit Model #####

    # Define models to try and their input parameters
    parameters = {"alpha": [10 ** i for i in range(-10, 3)]}
    GPReg = GaussianProcessRegressor()
    GPRegGS = GridSearchCV(GPReg, parameters, cv=5)  # search for optimal alpha value

    # Fit models and select the best trained one
    estimators = [GPRegGS]  # list of model types to try
    best_model = fit_model(
        estimators, df_all, features, targets[0], test_size, n_trials
    )  # fit on the main target
    # Plot partial dependences
    index = [i for i in range(n_features)]
    fig, ax = plt.subplots(figsize=(20, 6))
    plot_partial_dependence(
        best_model,
        df_all[features],
        index,
        features,
        targets[0],
        n_cols=n_features,
        ax=ax,
    )

    ##### 3.Bayesian optimization over the best model to maximize cycle life #####

    def objective_function(X):
        predicted_cycle = best_model.predict(X)[0]
        return -predicted_cycle  # negative cycles to be MINIMIZED

    # Bounds for each component mass fractions
    bounds = [
        {"name": "comp1", "type": "continuous", "domain": (0, 1)},
        {"name": "comp2", "type": "continuous", "domain": (0, 1)},
        {"name": "comp3", "type": "continuous", "domain": (0, 1)},
        {"name": "comp4", "type": "continuous", "domain": (0, 1)},
        {"name": "comp5", "type": "continuous", "domain": (0, 1)},
        {"name": "comp6", "type": "continuous", "domain": (0, 1)},
        {"name": "comp7", "type": "continuous", "domain": (0, 1)},
    ]

    # These constraints force the compositions to almost add up to 1
    constraints = [
        {"name": "constr_1", "constraint": "0.9 - np.sum(x,axis=1)"},  # >= 0.9
        {"name": "constr_2", "constraint": "np.sum(x,axis=1) - 1.0"},  # <= 1 
    ] 

    print("Running Bayesian optimization...")
    optimizer = bayesian_optimize(
        objective_function, bounds, constraints, max_time, max_iter, tolerance
    )
    optimizer.plot_convergence(results_dir + "/convergence.pdf")
    print("Finished!")

    ##### 4. Recommend best formulations #####

    # Select best compositions
    index = np.argsort(optimizer.Y.flatten())[range(n_rec)]
    recommended_compositions = optimizer.X[index]
    recommended_targets = -optimizer.Y.flatten()[index]
    formula_name = ["F_pred-{}".format(i) for i in range(n_rec)]

    print("Top {} predicted number of cycles: \n {}".format(n_rec, recommended_targets))

    # Create prediction dataframe -  don't forget to add back the dropped component (Additive-B)
    last_component = np.array([1 - recommended_compositions.sum(axis=1)]).T
    recommended_compositions_all = np.append(
        recommended_compositions, last_component, axis=1
    )
    all_features = features + drop
    assert all_features == ingredients

    df_prediction = pd.DataFrame(index=index, columns=column_names)
    df_prediction["Formulation"] = formula_name
    df_prediction[all_features] = recommended_compositions_all
    df_prediction[targets[0]] = recommended_targets

    if n_targets > 1:  # fit on other remaining targets
        for i in range(1, n_targets):
            other_model = fit_model(
                estimators, df_all, features, targets[i], test_size, n_trials
            )
            df_prediction[targets[i]] = other_model.predict(recommended_compositions)

    df_prediction.reset_index(inplace=True, drop=True)

    df_prediction.to_csv(results_dir + "/" + result_fname)

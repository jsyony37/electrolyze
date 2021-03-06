electrolyze
=======

**electrolyze** is a tool for recommending the optimal electrolyte compositions
for battery number of cycles. It performs machine learning model fitting followed
by Bayesian optimization of the best-fit model to search the complex composition
space for optimal battery life. The default model used is Gaussian process, but
any user-desired model can be added to the contest for best fit.

**electrolyze** is written in Python, and is interfaced with **scikit-learn**
for machine learning and **GPyOpt** for Bayesian optimization. Useful functions
are provided in `utiliities.py`. Here is a snippet of how to use the functions: 

.. code-block:: python

   best_model = fit_model(list_of_models, df, feature_names, target_name, n_fitting_trials)
   optimizer = bayesian_optimize(objective_function, bounds, constraints, max_time, max_iter, tolerance)

The driver `electrolyze` is a quick on-the-fly model-fitting and recommendation
tool that can be run directly on the commandline, and can be called as ::

  electrolyze

It reads input file by the name of `input_electrolyze`. A longer interactive workflow
with more extensive data exploration and analysis is provided in `workflow.ipynb`. 

Top-10 recommended formula are given in `best_recommendations.csv` file in the main
directory. Running `electrolyze` will deposit new results in a new directory.

Installation
------------

**electrolyze** requires python3.6 at minimum and can be installed via ::

  pip install -e .

or ::

  python setup.py install

		

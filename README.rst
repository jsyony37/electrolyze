electrolyze
=======

**electrolyze** is a tool for predicting the optimal compositions for
battery number of cycles. It uses Gaussian process regression followed
by Bayesian optimization of the best-fit model to search the complex
composition space for optimal battery life.

**electrolyze** is written in Python, and is interfaced with **scikit-learn**
for machine learning and **GPyOpt** for Bayesian optimization. Useful functions
are provided in `utiliities.py`.

Here is a snippet of how to use the functions: 

.. code-block:: python

   best_model = fit_model(list_of_models, df, feature_names, target_name, n_fitting_trials)
   optimizer = bayesian_optimize(objective, bounds, constraints, max_time, max_iter, tolerance)

The driver `electrolytes/main_run` is a quick on-the-fly model fitting and prediction tool
that can be run on the commandline. A longer interactive workflow with additional data exploration
and analysis is given in `workflow.ipynb`.
		
Installation
------------

**electrolyze** can be installed via ::

  python setup.py install

or ::

  pip install -e .

		

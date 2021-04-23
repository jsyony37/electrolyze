import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from argparse import ArgumentParser
from configparser import ConfigParser


def read_input_file():
    """ Reads electrolyze input file """
    parser = ArgumentParser()
    parser.add_argument("--file", "-f", help="settings file. Default input_electrolyze", default='input_electrolyze')
    options = parser.parse_args()
    inputs = ConfigParser()
    if not os.path.exists(options.file):
        print("ERROR: settings file {} not found".format(options.file))
        exit(-1)
    inputs.read(options.file)

    return options, inputs


def pearson(X,Y):
    """
    Attains Pearson correlations between features in X and the target Y
    
    Parameters:
    -----------
    X: pandas.DataFrame
        Feature matrix
    Y: pandas.Series
        Target values
    
    Returns:
    -----------
    corrcoeff: Pearson correlation coefficients
    """

    features = X.columns
    n_features = len(X)
    corrcoeff = list()
    pvalue = list()
    for idx, f in enumerate(features) : # loop over features  
        a, b = sp.stats.pearsonr(X[f],Y)
        corrcoeff.append(a)
        pvalue.append(b)
    
    return corrcoeff



def featurize(df,features,target,normalize=None):
    """
    Gets features from input dataframe
    
    Parameters:
    -----------
    df: pandas.DataFrame
        Input dataset including features and target
    features: list
        Feature column name(s)
    target: str
        Target column name
    normalize: str
        sklearn normalization method, e.g. standard, minmax
    
    Returns:
    -----------
    X : Pandas dataframe
        Fitting matrix of selected features
    Y : Pandas dataframe
        Target values     
    """
    from sklearn import preprocessing
            
    Y = np.asarray(df[target])
    X = np.asarray(df[features])
    # Normalize
    if normalize is None:
        pass
    elif normalize.lower() == 'standard':
        X = preprocessing.StandardScaler().fit(X).transform(X)
    elif normalize.lower() == 'minmax':
        X = preprocessing.MinMaxScaler().fit(X).transform(X)
    else:
        pass
    return X, Y



def fit_model(estimators,df,features,target,test_size,n_trials):
    """
    Fits all models in the input list of estimators and returns the best one
    
    Parameters:
    -----------
    estimators: list
        All estimator models to try for fitting
    df: pandas.Dataframe
        
    features: list
        Feature column name(s)
    target: str
        Target column name
    test_size: float
        A number between 0 and 1 indicating the proportion of the dataset
        to be set aside for testing fitted models
    n_trials: int
        Number of repetitions for fitting a given model
    
    Returns:
    -----------
    best_est: 
        The best estimator by testing error   
    """
    from sklearn.model_selection import train_test_split

    print('Fitting for {}...'.format(target))
    cv_scores = []
    errors = []
    y_hats = []
    y_tests = []
    all_estimators = []
    for normalize in [None]:
        X, y = featurize(df,features,target,normalize)
        for trial in range(n_trials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 
            for est in estimators:
                est.fit(X_train,y_train) 
                y_hat = est.predict(X_test)
                y_hats.append(y_hat)
                y_tests.append(y_test)
                cv_scores.append(est.best_score_)
                error = test_error(y_hat,y_test)
                errors.append(error)
                all_estimators.append(est)
    errors = np.array(errors)
    best_ind = np.where(errors == np.min(errors))[0][0] # select the lowest-error model index
    print('Error:',errors[best_ind])
    best_est = all_estimators[best_ind]
    
    # Plot testing vs predicted data alongside 1-to-1 line
    x = np.linspace(0,np.max(df[target]))
    y = x
    sns.set_context('talk')
    fig, ax = plt.subplots(1,figsize=(5,5))
    ax.scatter(y_tests[best_ind],y_hats[best_ind])
    ax.plot(y,x,'black')
    ax.set_ylabel('Predicted {}'.format('Number of Cycles' if target=='Measurement-3' else target))
    ax.set_xlabel('Actual {}'.format('Number of Cycles' if target=='Measurement-3' else target))
    ax.set_title('{} - Actual VS Predicted'.format(target))
    fig.savefig('best_fit_test.pdf')
    
    return best_est



def test_error(y_hat,y_test):
    """
    Computes average test error between predicted and actual target values
    
    Parameters:
    -----------
    y_hat: np.ndarray
        Predicted target quantities
    y_test: np.ndarray

    
    Returns:
    -----------
    error : float
        Average test error     
    """
    error = np.mean(abs((y_hat-y_test)/y_test))
    return error



def bayesian_optimize(objective_func,bounds,constraints,max_time,max_iter,tolerance):
    """
    Sets up and performs Bayesian optimization using GPyOpt
    Uses the expected improvement (EI) acquisition function, but can be changed
    Uses Gaussian process as the surrogate model, but can be changed

    Parameters:
    -----------
    objective_func: function
        The function to be minimized
    bounds: list
        Boundary conditions for each variable
    constraints: list
        Constraints that the variables must obey, in the <= 1 format
    max_time: float
        Maximum time to spend optimizing, upon reaching which the optimization finishes
    max_iter: int
        Maximum number of iterations, upon reaching which optimization finishes
    tolerence: float
        
        
    Returns:
    -----------
    optimizer: GPyOpt.methods.ModularBayesianOptimization
        The optimization object containing points of exploration 
    """ 
    import GPyOpt
    # Define the region where sampling is to be done
    feasible_region = GPyOpt.Design_space(space=bounds, constraints=constraints) 
    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)
    # Transform objective function into a GPyOpt object
    objective = GPyOpt.core.task.SingleObjective(objective_func)
    # Select model type (Gaussian process)
    bo_model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)
    # Select acquisition optimizer type
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    acquisition = GPyOpt.acquisitions.AcquisitionEI(bo_model, feasible_region, optimizer=aquisition_optimizer)
    # Select collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    # Complete optimizer object
    optimizer = GPyOpt.methods.ModularBayesianOptimization(bo_model, feasible_region, objective, 
                                                       acquisition, evaluator, initial_design)
    
    # Run optimization
    optimizer.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False) 
    
    return optimizer

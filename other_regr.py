
import datetime
import itertools
import math

import numpy as np
import pydotplus

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals.joblib import Parallel, cpu_count, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm

import gputils


# run other regression algorithms
def GradientTrees(seed):
    return GradientBoostingRegressor(n_estimators = 400, random_state=seed, max_depth=3)

def GaussProcess(seed):
    return GaussianProcessRegressor(random_state=seed)

def Linear(seed):
    return LinearRegression(n_jobs=1)

def Polynomial2(seed):
    return Polynomial(seed, 2)

def Polynomial3(seed):
    return Polynomial(seed, 3)

def Polynomial4(seed):
    return Polynomial(seed, 4)

def Polynomial5(seed):
    return Polynomial(seed, 5)

def Polynomial(seed, degree):
    return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False, n_jobs=1))

def DeepMLP800x4(seed):
    return MLPRegressor(hidden_layer_sizes=(800,800,800,800), alpha=1e-7, activation='relu',solver='lbfgs', max_iter=200, tol=1e-8, random_state=seed)

def DeepMLP200x4(seed):
    return MLPRegressor(hidden_layer_sizes=(200,200,200,200), alpha=1e-7, activation='relu',solver='lbfgs', max_iter=200, tol=1e-8, random_state=seed)

def DeepMLP50x15(seed):
    return MLPRegressor(hidden_layer_sizes=(50,50,50,50,50,50,50,50,50,50,50,50,50,50,50), alpha=1e-7, activation='relu',solver='lbfgs', max_iter=200, tol=1e-8, random_state=seed)

def SVRegression(seed):
    return svm.SVR(gamma='scale')


def parallel_fit(func, seeds, X=None, y=None, test_size=None, X_train=None, X_test=None, y_train=None, y_test=None):
    rmses_train = []
    rmses_test = []
    scores_train = []
    scores_test = []
    for seed in seeds:
        if X is not None and y is not None and test_size is not None:
            # each run a different split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        est = func(seed).fit(X_train, y_train)

        rmses_test.append(math.sqrt(mean_squared_error(est.predict(X_test), y_test)))
        scores_test.append(est.score(X_test, y_test))

        rmses_train.append(math.sqrt(mean_squared_error(est.predict(X_train), y_train)))
        scores_train.append(est.score(X_train, y_train))
        
    return rmses_test, scores_test, rmses_train, scores_train

if __name__ == "__main__":
    
    cv_size = 31
    n_jobs = 3

    # is normally calculated by joblib.Parallel, but we need it upfront to divide the seeds accordingly
    if n_jobs < 0:
        # inverse definition e.g. -2 = all cpu's but one
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    else:
        n_jobs = n_jobs

    n_jobs = min(n_jobs, cv_size)

    seeds_per_job = (cv_size // n_jobs) * np.ones(n_jobs, dtype=np.int)     # partition seeds
    seeds_per_job[:cv_size % n_jobs] += 1   # partition the remaining seeds over all jobs
    print('Jobs per CPU: {}'.format(seeds_per_job))
    seeds_per_job = [0] + np.cumsum(seeds_per_job).tolist()    # get the sums i.e. the actual indices

    # define data or data names
    datasets = [
    "f2_alt_n2000_python",
    "f1_n2000_python"]
    funcs = [Linear,
    Polynomial4,
    GradientTrees,
    GaussProcess,
    DeepMLP200x4]
    random_state = np.random.RandomState(6619927)
    seeds = random_state.randint(np.iinfo(np.int32).max, size=cv_size)
    
    for dataset in datasets:
        # read data and split x,y columns
        my_data = np.genfromtxt("data/" + dataset + ".csv", delimiter=',')
        X = my_data[:,:-1]
        y = my_data[:,-1]
        test_size = 0.5
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        gplogger_run = gputils.Mylogger(folder="runs/other_regr", fileprefix=dataset+"__", filesuffix="eval")
        gplogger_run.start()
        print('{:<11} {:<14} {:<11} {:<14} {:<15} {:<18} {:<15} {:<18} {:<30}'.format('~|R2@Train','IQR|R2@Train', '~|R2@Test','IQR|R2@Test', '~|RMSE@Train', 'IQR|RMSE@Train', '~|RMSE@Test', 'IQR|RMSE@Test', 'Config'))
        gplogger_run.stop()
        for func in funcs:
            result = Parallel(n_jobs=n_jobs, verbose=0)(
                        delayed(parallel_fit)(func, 
                                                seeds[seeds_per_job[i]:seeds_per_job[i+1]], 
                                                X_train=X_train, 
                                                X_test=X_test, 
                                                y_train=y_train, 
                                                y_test=y_test) for i in range(n_jobs))
            rmses_test, scores_test, rmses_train, scores_train = zip(*result)
            rmses_test = list(itertools.chain.from_iterable(rmses_test))
            scores_test = list(itertools.chain.from_iterable(scores_test))
            rmses_train = list(itertools.chain.from_iterable(rmses_train))
            scores_train = list(itertools.chain.from_iterable(scores_train))

            gplogger_run.start()
            print('{:<11.3f} {:<14.5f} {:<11.3f} {:<14.5f} {:<15.3e} {:<18.3e} {:<15.3e} {:<18.3e} {:<30}'.format(np.median(scores_train), np.subtract(*np.percentile(scores_train, [75, 25])), np.median(scores_test), np.subtract(*np.percentile(scores_test, [75, 25])), np.median(rmses_train), np.subtract(*np.percentile(rmses_train, [75, 25])),np.median(rmses_test), np.subtract(*np.percentile(rmses_test, [75, 25])), func.__name__))
            gplogger_run.stop()
            
            gplogger_cv_sum = gputils.Mylogger(folder="runs/other_regr/summed", fileprefix=dataset+"__"+func.__name__, filesuffix="eval")
            gplogger_cv_sum.start()
            print('{:<10} {:<10} {:<14} {:<14}'.format('R2@Train','R2@Test', 'RMSE@Train', 'RMSE@Test'))
            gplogger_cv_sum.stop()
            gplogger_cv_sum_csv = gputils.Mylogger(folder="runs/other_regr/csv", fileprefix=dataset+"__"+func.__name__, filesuffix="eval")

            for i in range(len(scores_train)):
                gplogger_cv_sum.start()
                print('{:<10.3f} {:<10.3f} {:<14.3e} {:<14.3e}'.format(scores_train[i], scores_test[i], rmses_train[i], rmses_test[i]))
                gplogger_cv_sum.stop()

                
                gplogger_cv_sum_csv.start()
                print('{},{},{},{}'.format(scores_train[i], scores_test[i], rmses_train[i], rmses_test[i]))
                gplogger_cv_sum_csv.stop()
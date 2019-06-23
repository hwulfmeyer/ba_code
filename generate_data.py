
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import normalize, scale
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import math
import gputils

def func1(X):
    a = X[:,0]
    b = X[:,1]
    c = X[:,2]
    d = X[:,3]
    t1opt = 0.5 + 0.2*a + 0.3*b - 0.2*(a*b)
    t2opt = 1 - 0.3*c - 0.1*c**2
    t3opt = 0.1*(1 - (d - 0.5)**2)
    topt = t1opt * t2opt + t3opt
    return t1opt, t2opt, t3opt, topt

def func2(X):
    a = X[:,0]
    b = X[:,1]
    return np.cos(2*a) + np.sin(3*b)

def func2_alt(a=1,b=1):
    return 0.9*np.cos(2.1*a) + 1.1*np.sin(3.1*b)

def gendata(filename="f1_n2000_python_alternative", data_size = 2000, seed=544112):
    rstate = np.random.RandomState(seed)
    X = rstate.rand(data_size, 4)
    X = np.around(X, 18)
    #Y = func2_alt(X)
    x0,x1,x2,Y = func1(X)
    Y = [[y] for y in Y]
    x0 = [[x] for x in x0]
    x1 = [[x] for x in x1]
    x2 = [[x] for x in x2]
    Y = np.around(Y, 18)
    x0 = np.around(x0, 18)
    x1 = np.around(x1, 18)
    x2 = np.around(x2, 18)
    my_data = np.append(x0, x1, axis=1)
    my_data = np.append(my_data, x2, axis=1)
    my_data = np.append(my_data, Y, axis=1)
    np.savetxt("data/" + filename +".csv", my_data, delimiter=",", fmt='%1.18f')
    testdata(filename=filename)
    return my_data

def gendata_noisy(filename="f1_noisy_1p_n2000_python", data_size = 2000, seed=544112):
    rstate = np.random.RandomState(seed)
    X = rstate.rand(data_size, 4)
    X = np.around(X, 18)
    #_,_,_,Y = func1(X)
    Y = func2(X)
    noise = rstate.normal(0,0.05,data_size)
    print(gputils.rmse(y=Y, y_pred=Y+noise))
    print(gputils.mpe(y=Y, y_pred=Y+noise))
    Y = Y + noise
    Y = [[y] for y in Y]
    Y = np.around(Y, 18)
    my_data = np.append(X,Y, axis=1)
    #np.savetxt("data/" + filename +".csv", my_data, delimiter=",", fmt='%1.18f')
    #testdata(filename=filename)
    return my_data


def testdata(filename="f2_n2000_python"):
    my_data = np.genfromtxt("data/" + filename +".csv", delimiter=',')
    X = my_data[:,[0,1]]
    y_a = [[y] for y in my_data[:,-1]]
    y_b = [[y] for y in func2_alt(X[:,0], X[:,1])]
    y_b = np.around(y_b, 18)
    y = np.append(y_a,y_b, axis=1)
    y_c = [[j[0] == j[1]] for j in y]
    y = np.append(y,y_c, axis=1)
    for j in y:
        if j[2] ==  0:
            print("ERROR")


def normalizedata(filename="PXD007612_clean"):
    my_data = np.genfromtxt("data/" + filename +".csv", delimiter=',')
    my_data = normalize(my_data, norm='max', axis=0)
    np.savetxt("data/" + filename + "_max-norm.csv", my_data, delimiter=",", fmt='%1.18f')


def standardizedata(filename="PXD007612_clean"):
    my_data = np.genfromtxt("data/" + filename +".csv", delimiter=',')
    my_data = scale(my_data, with_mean=False, axis=0)
    np.savetxt("data/" + filename + "_std_nocenter.csv", my_data, delimiter=",", fmt='%1.18f')


def featureselection(est, filename="PXD007612_clean"):
    my_data = np.genfromtxt("data/" + filename + ".csv", delimiter=',')
    X = my_data[:,:-1]
    y = my_data[:,-1]
    selector = RFECV(est, step=1, min_features_to_select=5, cv=3, n_jobs=4)
    selector = selector.fit(X, y)
    for i,x in enumerate(selector.support_):
        if x == True:
            print("X{}".format(i))
    print(selector.ranking_)
    
    
if __name__ == "__main__":
    #my_data = gendata()
    gendata_noisy()
    #est = GradientBoostingClassifier(n_estimators = 300)
    #featureselection(est)

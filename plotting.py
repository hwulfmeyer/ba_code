import datetime
import math
import pickle
import time
from copy import deepcopy


import gplearn.fitness
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import scipy as sp
import seaborn as sns
import sympy
from gplearn.genetic import SymbolicRegressor
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from numpy import genfromtxt
from scipy.stats import binom
from statsmodels.sandbox.stats.multicomp import multipletests

import gputils

my_data = genfromtxt('data/f1_n200_python.csv', delimiter=',')
data_in = my_data[:,:-1]
data_out = my_data[:,-1]


def func1(a=1,b=1,c=1,d=1):
    # vom 17-12-2018_04:06:31
    return (0.0463*a*(a + c) - 0.0388*b*(a*c + 2.7335*a*(a + c) + c**2*(a + c)) - 0.0794*(a + c)*(d - 0.508)**2 + (a + c)*(-0.0388*a*(c**2 - 0.597) + 0.0776*a + 0.2746*b - 0.0388*c**2 - 0.197*c - 0.0388*(a + c)*(b - 0.597) + 0.6114))/(a + c)
    
# 0, 1, 2, 3
def func12(a=1,b=1,c=1,d=1):
    t1opt = 0.5 + 0.2*a + 0.3*b - 0.2*(a*b)
    t2opt = 1 - 0.3*c - 0.1*c**2
    t3opt = 0.1*(1 - (d - 0.5)**2)
    topt = t1opt * t2opt + t3opt
    return t1opt, t2opt, t3opt, topt

def func3(a=1,b=1):
    return np.cos(2*a) + np.sin(3*b)


def func3_alter(a=1,b=1):
    return 0.9*np.cos(2.1*a) + 1.1*np.sin(3.1*b)

def plotFunc2():
    u = np.linspace(0, 1, 400)
    v = u
    U, V = np.meshgrid(u, v)
    sns.set(style="whitegrid")
    t1,t2,t3,Y = func12(a=U,b=V,c=u,d=u)
    fig = plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.contour3D(U, V, t1, 75, cmap='binary', linewidths=1)
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('Y_a')
    plt.savefig("plotting/plots/func1a.PNG", dpi=300)
    fig.show()

    fig = plt.figure(2)
    plt.plot(u,t2)
    plt.xlabel('x2')
    plt.ylabel('Y_b')
    plt.savefig("plotting/plots/func1b.PNG", dpi=300)

    fig = plt.figure(3)
    plt.plot(v,t3)
    plt.xlabel('x3')
    plt.ylabel('Y_c')
    plt.savefig("plotting/plots/func1c.PNG", dpi=300)
    plt.show()

    input()

def plotFunc3():
    u = np.linspace(0, 1, 200)
    v = u
    U, V = np.meshgrid(u, v)

    Y = func3(U,V)
    Y_alt = func3_alter(U,V)

    fig1 = plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.contour3D(U, V, Y, 75, cmap='binary', linewidths=1)
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('y')
    fig1.show()
    plt.savefig("plotting/plots/func2.PNG", dpi=300)

    fig2 = plt.figure(2)
    ax1 = plt.axes(projection='3d')
    ax1.contour3D(U, V, Y_alt, 75, cmap='binary', linewidths=1)
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('y')
    fig2.show()
    plt.savefig("plotting/plots/func2_new.PNG", dpi=300)
    
    fig3 = plt.figure(3)
    ax1 = plt.axes(projection='3d')
    ax1.contour3D(U, V, Y_alt-Y, 75, cmap='binary', linewidths=1)
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('y')
    fig3.show()
    input()

def plotError():
    u = np.linspace(0, 1, 3000)
    v = u
    w = u
    x = u

    U, V = np.meshgrid(u, v)

    Y1 = func1(U, V, U, V)# - func2(U, V, .5, .5)

    fig1 = plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.contour3D(u, v, Y1, 50, cmap='binary')
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('y_Error')
    fig1.show()
    #plt.savefig("plots/y_Error_x0-x1.PNG", dpi=300)


    Y2 = func1(V, U, V, U)# - func2(.5, U, V, .5)

    fig2 = plt.figure(2)
    ax2 = plt.axes(projection='3d')
    ax2.contour3D(u, v, Y2, 50, cmap='binary')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('y_Error')
    fig2.show()
    #plt.savefig("plots/y_Error_x1-x2.PNG", dpi=300)

    Y3 = func1(V, U, U, V)# - func2(.5, .5, U, V)

    fig3 = plt.figure(3)
    ax3 = plt.axes(projection='3d')
    ax3.contour3D(u, v, Y3, 50, cmap='binary')
    ax3.set_xlabel('x2')
    ax3.set_ylabel('x3')
    ax3.set_zlabel('y_Error')
    fig3.show()
    #plt.savefig("plots/y_Error_x2-x3.PNG", dpi=300)

    fig4 = plt.figure(4)
    hist, bins = np.histogram(data_out, bins=10)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax3 = plt.bar(center, hist, align='center', width=width)
    fig4.show()
    #plt.savefig("plots/y_histogram_10bins.PNG", dpi=300)
    input()

# plot bins of Y data
def y_bins():
    fig = plt.figure(1)
    n, bins, patches = plt.hist(data_out, bins=20, density=False)
    #plt.axis([-0.05, 3.05, 0, 1.15])
    plt.grid(True)
    plt.savefig("plotting/plots/func1_bins.PNG", dpi=300)
    plt.show()
    input()


def x_bins(dim):
    fig = plt.figure(1)
    n, bins, patches = plt.hist(data_in[:,dim], bins=20, density=True)
    #plt.axis([-0.05, 3.05, 0, 1.15])
    plt.grid(True)
    plt.show()
    #input()

def plot_gpmodeldata(diruri):
    from os import listdir
    from os.path import isfile, join
    func = "f1"
    onlyfiles = [f for f in listdir(diruri) if isfile(join(diruri, f)) and func in f]
    onlyfiles.sort()
    est_gp_pickles = []
    for gpmodel in onlyfiles:
        print(gpmodel)
        with open(diruri + gpmodel, 'rb') as f:
            est_gp = pickle.load(f)
            est_gp_pickles.append(est_gp)
    
    dfframes = []
    for gppickle in est_gp_pickles:
        df = pd.DataFrame(gppickle.run_details_["best_fitness"], columns=["Fitness"])
        dfframes.append(df)

    df = pd.concat(dfframes, axis=1, ignore_index=False)
    #df = df.reset_index()
    df.columns = ["PGP Kommenda", "PGP Length"]
    print(df)
    #ax = df.plot(x='Crossover', kind='line', style=['-', '--', '-.', '.-'], legend=False, xticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax = df.plot(kind='line', style=['-', '--', '-.', '.-'], logy=True)
    ax.set_xlabel("Generation")
    ax.set_ylabel("RMSE")

    fig = ax.get_figure()
    plt.savefig("plotting/plots/generations_f1.PNG", dpi=500)
    plt.show()
    """
    index = 1
    gensteps = [0,25,150,800,1249]
    fronts_raw = []
    fronts = est_gp_pickles[index]._paretofronts
    for i in gensteps:
        for prg in fronts[i]:
            fronts_raw.append([prg.raw_fitness_, prg.complexity_, i])
    complexity = ("Kommenda" if index == 0 else "Length")
    df = pd.DataFrame(fronts_raw, columns=["RMSE", complexity, "Generation"])

    #ax = sns.scatterplot(x="Fitness", y="Complexity", style="Generation", data=df, alpha=0.7, legend="brief")
    g = sns.lmplot(x="RMSE", y=complexity, data=df, hue='Generation', fit_reg=False, markers=['*', 'x', '+', '1', 'o'], legend=False, height=4, aspect=1.3)
    #g = g.set(xlim=(1e-02 * 0.9, 3))    # f1 kommenda, length
    #g = g.set(ylim=(1, 10e06*0.5))      # f1 kommenda
    plt.xscale('log')
    plt.yscale('log')
    if func == "f2":
        plt.legend(loc='best')
    plt.savefig("plotting/plots/" + func + "_pgp_" + complexity + "_fronts.PNG", dpi=500)
    plt.show()"""

    

def plot_csvruns():
    diruri = "/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/algorithms/first/medians/CSV/"
    func = "f2"
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(diruri) if isfile(join(diruri, f)) and func in f]
    onlyfiles.sort()
    dframes = []
    for p in onlyfiles:
        print(p)
        df = pd.read_csv(diruri + p)
        df.columns = ["Gen", "LengthA",  "KommendaA",   "FitnessA",   "LengthB",  "KommendaB",  "FitnessB",   "Time Left",  "Front Size"]
        dframes.append(df["FitnessB"])

    df = pd.concat(dframes, axis=1, ignore_index=False)
    #df = df.reset_index()
    df.columns = ["EplexGP", "StandardGP", "PGP Kommenda", "PGP Length"]
    print(df)
    #ax = df.plot(x='Crossover', kind='line', style=['-', '--', '-.', '.-'], legend=False, xticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ylim = (0.0079,0.11) if func == "f1" else (0.016,0.4)
    ax = df.plot(kind='line', style=['-', '--', ':', '-.'], logy=True, xlim=(0,1250), ylim = ylim, figsize=(10, 4))    #6.4, 4,8
    ax.set_xlabel("Generation")
    ax.set_ylabel("RMSE")

    plt.savefig("plotting/plots/generations_f2.PNG", dpi=500)
    plt.show()



def boxplots(filename):
    my_data = np.genfromtxt("plotting/" + filename, delimiter=',')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    all_data = [my_data[:,i] for i in range(my_data.shape[1])]
    # notch shape box plot
    bplot2 = ax.boxplot(all_data,
                            notch=False,
                            vert=True,
                            whis=1.5,
                            sym='+')

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))], )
    ax.set_xlabel('Parsimony')
    ax.set_ylabel('RMSE@Test')

    # add x-tick labels
    plt.setp(ax, xticks=[y+1 for y in range(len(all_data))],
            xticklabels=['2e-05','3e-05','4e-05','5e-05','6e-05','7e-05',])
            #xticklabels=['1e-08', '2e-08', '3e-08', '4e-08', '5e-08', '6e-08', '7e-08', '8e-08', '9e-08', '1e-07'])

    plt.show()

def significancetest(diruri, testsize, num_files, row_desc):
    # testsize = number of seperate tests to do
    # num_files = files per test
    # row_desc = list of the numbers representing the paramter (do not set to string/char)
    columns = [2,3,4,5]
    num_columns = len(columns)

    row_desc= np.array(row_desc)
    row_desc = row_desc.reshape((row_desc.shape[0],1))

    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(diruri) if isfile(join(diruri, f)) ] # and 'f2' in f]
    onlyfiles.sort()
    fileslength = len(onlyfiles)/testsize
    if num_files != fileslength:
        raise Exception('Wrong number of num_files or the testsize! ' + str(num_files) + ' vs ' + str(fileslength))
    file_lists = []
    for _ in range(testsize):
        files = []
        for _ in range(int(fileslength)):
            files.append(onlyfiles.pop())
        file_lists.append(files)
    
    np.set_printoptions(precision=2)
    # nur für die ersten num_files dateien
    median_iqr_all = np.zeros((num_files,2*testsize))
    for j, files in enumerate(file_lists):
        for ttt in files:
            print(ttt)

        # extract data vom csv files
        # num_files files, 31 rows and 4 columns per files (the compared data)
        files_data = np.zeros((num_files,31,num_columns))
        for i,x in enumerate(files):
            # rmse@train, rmse@test, length, kommenda
            files_data[i] = genfromtxt(diruri + x, delimiter=',')[:,columns]

        # num_files files, median and iqr (2) for all num_columns columns
        median_iqr = np.zeros((num_files,2,num_columns))
        for i,median_iqr_file in enumerate(median_iqr):
            # calc median and iqr
            for k in range(len(median_iqr_file[0])):
                x = files_data[i][:,k]
                iqr = np.subtract(*np.percentile(x, [75, 25]))
                median = np.median(x)
                median_iqr_file[0][k] = median
                median_iqr_file[1][k] = iqr

        #best of medians of RMSE@Test
        # 0 => [0,1]
        # 1 => [2,3]
        median_iqr_all[:,j*2] = median_iqr[:,0][:,1]    # median    rmse@test
        median_iqr_all[:,j*2+1] = median_iqr[:,1][:,1]  # iqr       rmse@test
        best_file = np.argmin(median_iqr[:,0][:,1])

        # significance test
        testresult = np.zeros((num_files,num_columns))    # num_files
        for i in range(testresult.shape[0]):
            for k in range(testresult.shape[1]):
                x = files_data[best_file][:,k]
                y = files_data[i][:,k]
                if i != best_file:
                    _,pvalue = sp.stats.mannwhitneyu(x,y,alternative='two-sided')
                    testresult[i][k] = pvalue
                else:
                    testresult[i][k] = 0
        #print(testresult)
        for k in range(testresult.shape[1]):
            #http://www.statsmodels.org/0.8.0/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
            _, testresult[:,k] ,_ ,_ = multipletests(testresult[:,k], alpha=0.05, method='holm')   # holm-bonferroni
            """
            It gets more likelier to have a false positive the more tests we run
            """
        print()
        print(files[best_file])
        testresult = np.append(row_desc, testresult, axis=1)
        np.set_printoptions(precision=2)
        x_str = np.array_repr(testresult)
        x_str = x_str.replace('\n        ', '')
        x_str = x_str.replace('array([[', '')
        x_str = x_str.replace(']])', '\\\\')
        x_str = x_str.replace('],\n       [', '\\\\\n')
        x_str = x_str.replace(', ', ' & ')
        print(x_str)
        print("\n")
        """print("Median & IQR")
        np.set_printoptions(precision=2)
        median_iqr_output = np.append(row_desc, median_iqr_all[:,[j*2, j*2+1]], axis=1)
        x_str = np.array_repr(median_iqr_output)
        x_str = x_str.replace('\n        ', '')
        x_str = x_str.replace('array([[', '')
        x_str = x_str.replace(']])', '\\\\')
        x_str = x_str.replace('],\n       [', '\\\\\n')
        x_str = x_str.replace(', ', ' & ')
        print(x_str)
        print("\n\n")"""
    print("Median & IQR for all")
    np.set_printoptions(precision=4)
    median_iqr_output = np.append(row_desc, median_iqr_all, axis=1)
    x_str = np.array_repr(median_iqr_output)
    x_str = x_str.replace('\n        ', '')
    x_str = x_str.replace('array([[', '')
    x_str = x_str.replace(']])', '\\\\')
    x_str = x_str.replace('],\n       [', '\\\\\n')
    x_str = x_str.replace(', ', ' & ')
    print(x_str)
    print("\n\n")


def signigicancetestpandas(diruri, testsize, num_files, row_desc, columnsToTest, reverse):
    # testsize = number of seperate tests to do
    # num_files = files per test
    # row_desc = list of the numbers representing the paramter (do not set to string/char)
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(diruri) if isfile(join(diruri, f)) and ('f1_n2000' in f or 'f2_alt' in f)] #and ('pc1.0' in f or 'pc0.0' in f)]
    onlyfiles.sort(reverse=reverse)
    for f in onlyfiles:
        print(f)
    fileslength = len(onlyfiles)/testsize
    print(onlyfiles)
    if num_files != fileslength:
        raise Exception('Wrong number of num_files or the testsize! ' + str(num_files) + ' vs ' + str(fileslength))
    dframes = []
    testResults = []
    for _ in range(testsize):
        file_lists = []
        files = []
        for _ in range(int(fileslength)):
            k = onlyfiles.pop()
            print(k)
            files.append(k)
        file_lists.append(files)
        filesData = []
        myList = []
        for p in files:
            df = pd.read_csv(diruri + p, header=None)
            df.columns = ["R2@TRain","R2@Test","RMSE@Train","RMSE@Test","Length", "Kommenda"]
            filesData.append(df[columnsToTest])
            df = df[columnsToTest].sort_values(by=[columnsToTest[0]])
            df = df.reset_index(drop=True)
            rmse = df[columnsToTest[0]].median(axis=0)
            Q1 = df[columnsToTest[0]].quantile(0.25)
            Q3 = df[columnsToTest[0]].quantile(0.75)
            rmseIQR = Q3 - Q1

            length = df[columnsToTest[1]].median(axis=0)    #df["Length"][15]
            Q1 = df[columnsToTest[1]].quantile(0.25)
            Q3 = df[columnsToTest[1]].quantile(0.75)
            lengthIQR = Q3 - Q1
            #print("rmse:{:<8.4f}  length:{:<8.0f}".format(rmse, length))
            myList.append([rmse, rmseIQR, length, lengthIQR])
        df = pd.DataFrame(myList, columns=[columnsToTest[0], "IQRa", columnsToTest[1], "IQRb"])
        #print(df)
        dframes.append(df)
        
        # significance test
        curTestResults = []
        if 'R2' in columnsToTest[0]:
            bestIdx = df[columnsToTest[0]].idxmax(axis=1)
        else:
            bestIdx = df[columnsToTest[0]].idxmin(axis=1)
        for row in range(0, len(myList)): # row
            rowResults = []
            if row == bestIdx:
                continue
            dfBest = filesData[bestIdx]
            dfRow = filesData[row]
            for col in range(0, 2): # column
                _,pvalue = sp.stats.mannwhitneyu(dfBest.iloc[:,col], dfRow.iloc[:,col], alternative='two-sided')
                rowResults.append(pvalue)
            curTestResults.append(rowResults)
        curTestResults = np.array(curTestResults)
        # holm-bonferri correction for multiple significance tests
        #for col in range(0, 2): # column
            #_, curTestResults[:,col] ,_ ,_ = multipletests(curTestResults[:,col], alpha=0.05, method='holm')
        
        testdf = pd.DataFrame(curTestResults, columns=columnsToTest)
        testResults.append(testdf)   #[["RMSE@Test", "Length"]])

    df = pd.concat(dframes, axis=1, ignore_index=False)
    print(df)
    #print(df[["RMSE@Test", "IQRa"]])

    print("\n\nSignificance Test Result:")
    for result in testResults:
        print(result)

    print("\n\n")
    for i, k in enumerate(row_desc):
        rowstring = "\multicolumn{1}{ |c|}{"
        if isinstance(k, str):
            rowstring = rowstring + "\small{" + k + "}}"
        else:
            rowstring = rowstring + "{:0.1f}".format(k) + "}"

        for j, frames in enumerate(dframes):
            
            if 'R2' in columnsToTest[0]:
                bestIdx = frames[columnsToTest[0]].idxmax(axis=1)
            else:
                bestIdx = frames[columnsToTest[0]].idxmin(axis=1)
                
            toAdd = "{:0.4f}".format(frames[columnsToTest[0]][i])[1:]
            if bestIdx == i:
                toAdd = "\\textbf{" + toAdd + "}" + "\hphantom{ *}"
            elif testResults[j][columnsToTest[0]][i-1 if i > bestIdx else i] < 1e-02:
                toAdd =  toAdd + " *"
            else:
                toAdd =  toAdd + "\hphantom{ *}"
            rowstring = rowstring + " & " + toAdd

            toAdd = "{:<0.4f}".format(frames["IQRa"][i])[1:]
            if bestIdx == i:
                toAdd = "\\textbf{" + toAdd + "}"
            rowstring = rowstring + " & " + toAdd

            toAdd = "{:0.0f}".format(frames[columnsToTest[1]][i])
            if bestIdx == i:
                toAdd = "\\textbf{" + toAdd + "}" + "\hphantom{ *}"
            elif testResults[j][columnsToTest[1]][i-1 if i > bestIdx else i] < 1e-02:
                toAdd =  toAdd + " *"
            else:
                toAdd =  toAdd + "\hphantom{ *}"
            rowstring = rowstring + " & " + toAdd

            toAdd = "{:<0.1f}".format(frames["IQRb"][i])
            if bestIdx == i:
                toAdd = "\\textbf{" + toAdd + "}"
            rowstring = rowstring + " & " + toAdd


        rowstring = rowstring + "\\\\"
        print(rowstring)


def signigicancetestpandas2(diruri, testsize, num_files, row_desc, columnsToTest, reverse):
    # testsize = number of seperate tests to do
    # num_files = files per test
    # row_desc = list of the numbers representing the paramter (do not set to string/char)
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(diruri) if isfile(join(diruri, f)) and '5p' in f] #and ('pc1.0' in f or 'pc0.0' in f)]
    onlyfiles.sort(reverse=reverse)
    for f in onlyfiles:
        print(f)
    fileslength = len(onlyfiles)/testsize
    print(onlyfiles)
    if num_files != fileslength:
        raise Exception('Wrong number of num_files or the testsize! ' + str(num_files) + ' vs ' + str(fileslength))
    dframes = []
    testResults = []
    for _ in range(testsize):
        file_lists = []
        files = []
        for _ in range(int(fileslength)):
            k = onlyfiles.pop()
            print(k)
            files.append(k)
        file_lists.append(files)
        filesData = []
        myList = []
        for p in files:
            df = pd.read_csv(diruri + p, header=None)
            df.columns = ["R2@Train","R2@Test","RMSE@Train","RMSE@Test","Length", "Kommenda"]
            filesData.append(df[columnsToTest])
            df = df[columnsToTest].sort_values(by=[columnsToTest[0]])
            df = df.reset_index(drop=True)
            rmseA = df[columnsToTest[0]].median(axis=0)
            Q1 = df[columnsToTest[0]].quantile(0.25)
            Q3 = df[columnsToTest[0]].quantile(0.75)
            rmseIQRA = Q3 - Q1

            rmseB = df[columnsToTest[1]].median(axis=0)
            Q1 = df[columnsToTest[1]].quantile(0.25)
            Q3 = df[columnsToTest[1]].quantile(0.75)
            rmseIQRB = Q3 - Q1

            length = df[columnsToTest[2]].median(axis=0)    #df["Length"][15]
            Q1 = df[columnsToTest[2]].quantile(0.25)
            Q3 = df[columnsToTest[2]].quantile(0.75)
            lengthIQR = Q3 - Q1
            #print("rmse:{:<8.4f}  length:{:<8.0f}".format(rmse, length))
            myList.append([rmseA, rmseIQRA, rmseB, rmseIQRB, length, lengthIQR])
        df = pd.DataFrame(myList, columns=[columnsToTest[0], "IQRa", columnsToTest[1], "IQRb", columnsToTest[2], "IQRc"])
        #print(df)
        dframes.append(df)
        
        # significance test
        curTestResults = []
        if 'R2' in columnsToTest[0]:
            bestIdx = df[columnsToTest[0]].idxmax(axis=1)
        else:
            bestIdx = df[columnsToTest[0]].idxmin(axis=1)
        for row in range(0, len(myList)): # row
            rowResults = []
            if row == bestIdx:
                continue
            dfBest = filesData[bestIdx]
            dfRow = filesData[row]
            for col in range(0, 3): # column
                _,pvalue = sp.stats.mannwhitneyu(dfBest.iloc[:,col], dfRow.iloc[:,col], alternative='two-sided')
                rowResults.append(pvalue)
            curTestResults.append(rowResults)
        curTestResults = np.array(curTestResults)
        # holm-bonferri correction for multiple significance tests
        #for col in range(0, 2): # column
            #_, curTestResults[:,col] ,_ ,_ = multipletests(curTestResults[:,col], alpha=0.05, method='holm')
        
        testdf = pd.DataFrame(curTestResults, columns=columnsToTest)
        testResults.append(testdf)   #[["RMSE@Test", "Length"]])

    df = pd.concat(dframes, axis=1, ignore_index=False)
    print(df)
    #print(df[["RMSE@Test", "IQRa"]])

    print("\n\nSignificance Test Result:")
    for result in testResults:
        print(result)

    print("\n\n")
    for i, k in enumerate(row_desc):
        rowstring = "\multicolumn{1}{ |c|}{"
        if isinstance(k, str):
            rowstring = rowstring + "\small{" + k + "}}"
        else:
            rowstring = rowstring + "{:0.1f}".format(k) + "}"

        #for j, frames in enumerate(dframes):
        frames = dframes[1]
        j = 1
        if 'R2' in columnsToTest[0]:
            bestIdx = frames[columnsToTest[0]].idxmax(axis=1)
        else:
            bestIdx = frames[columnsToTest[0]].idxmin(axis=1)
            
        toAdd = "{:0.4f}".format(frames[columnsToTest[0]][i])[1:]
        if bestIdx == i:
            toAdd = "\\textbf{" + toAdd + "}" + "\hphantom{ *}"
        elif testResults[j][columnsToTest[0]][i-1 if i > bestIdx else i] < 1e-02:
            toAdd =  toAdd + " *"
        else:
            toAdd =  toAdd + "\hphantom{ *}"
        rowstring = rowstring + " & " + toAdd

        toAdd = "{:<0.4f}".format(frames["IQRa"][i])[1:]
        if bestIdx == i:
            toAdd = "\\textbf{" + toAdd + "}"
        rowstring = rowstring + " & " + toAdd

        toAdd = "{:0.4f}".format(frames[columnsToTest[1]][i])[1:]
        if bestIdx == i:
            toAdd = "\\textbf{" + toAdd + "}" + "\hphantom{ *}"
        elif testResults[j][columnsToTest[1]][i-1 if i > bestIdx else i] < 1e-02:
            toAdd =  toAdd + " *"
        else:
            toAdd =  toAdd + "\hphantom{ *}"
        rowstring = rowstring + " & " + toAdd

        toAdd = "{:<0.4f}".format(frames["IQRb"][i])[1:]
        if bestIdx == i:
            toAdd = "\\textbf{" + toAdd + "}"
        rowstring = rowstring + " & " + toAdd

        toAdd = "{:0.0f}".format(frames[columnsToTest[2]][i])
        if bestIdx == i:
            toAdd = "\\textbf{" + toAdd + "}" + "\hphantom{ *}"
        elif testResults[j][columnsToTest[2]][i-1 if i > bestIdx else i] < 1e-02:
            toAdd =  toAdd + " *"
        else:
            toAdd =  toAdd + "\hphantom{ *}"
        rowstring = rowstring + " & " + toAdd

        toAdd = "{:<0.1f}".format(frames["IQRc"][i])
        if bestIdx == i:
            toAdd = "\\textbf{" + toAdd + "}"
        rowstring = rowstring + " & " + toAdd


        rowstring = rowstring + "\\\\"
        print(rowstring)

def signigicancetestpandas3(diruri, testsize, num_files, row_desc, columnsToTest, reverse):
    # testsize = number of seperate tests to do
    # num_files = files per test
    # row_desc = list of the numbers representing the paramter (do not set to string/char)
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(diruri) if isfile(join(diruri, f))] #and ('pc1.0' in f or 'pc0.0' in f)]
    onlyfiles.sort(reverse=reverse)
    for f in onlyfiles:
        print(f)
    fileslength = len(onlyfiles)/testsize
    print(onlyfiles)
    if num_files != fileslength:
        raise Exception('Wrong number of num_files or the testsize! ' + str(num_files) + ' vs ' + str(fileslength))
    dframes = []
    testResults = []
    for _ in range(testsize):
        file_lists = []
        files = []
        for _ in range(int(fileslength)):
            k = onlyfiles.pop()
            print(k)
            files.append(k)
        file_lists.append(files)
        filesData = []
        myList = []
        for p in files:
            df = pd.read_csv(diruri + p, header=None)
            df.columns = ["R2@Train","R2@Test","RMSE@Train","RMSE@Test"]
            filesData.append(df[columnsToTest])
            df = df[columnsToTest].sort_values(by=[columnsToTest[0]])
            df = df.reset_index(drop=True)
            rmseA = df[columnsToTest[0]].median(axis=0)
            Q1 = df[columnsToTest[0]].quantile(0.25)
            Q3 = df[columnsToTest[0]].quantile(0.75)
            rmseIQRA = Q3 - Q1

            rmseB = df[columnsToTest[1]].median(axis=0)
            Q1 = df[columnsToTest[1]].quantile(0.25)
            Q3 = df[columnsToTest[1]].quantile(0.75)
            rmseIQRB = Q3 - Q1

            #print("rmse:{:<8.4f}  length:{:<8.0f}".format(rmse, length))
            myList.append([rmseA, rmseIQRA, rmseB, rmseIQRB])
        df = pd.DataFrame(myList, columns=[columnsToTest[0], "IQRa", columnsToTest[1], "IQRb"])
        #print(df)
        dframes.append(df)
        
        # significance test
        curTestResults = []
        if 'R2' in columnsToTest[0]:
            bestIdx = df[columnsToTest[0]].idxmax(axis=1)
        else:
            bestIdx = df[columnsToTest[0]].idxmin(axis=1)
        for row in range(0, len(myList)): # row
            rowResults = []
            if row == bestIdx:
                continue
            dfBest = filesData[bestIdx]
            dfRow = filesData[row]
            for col in range(0, 2): # column
                _,pvalue = sp.stats.mannwhitneyu(dfBest.iloc[:,col], dfRow.iloc[:,col], alternative='two-sided')
                rowResults.append(pvalue)
            curTestResults.append(rowResults)
        curTestResults = np.array(curTestResults)
        # holm-bonferri correction for multiple significance tests
        #for col in range(0, 2): # column
            #_, curTestResults[:,col] ,_ ,_ = multipletests(curTestResults[:,col], alpha=0.05, method='holm')
        
        testdf = pd.DataFrame(curTestResults, columns=columnsToTest)
        testResults.append(testdf)   #[["RMSE@Test", "Length"]])

    df = pd.concat(dframes, axis=1, ignore_index=False)
    print(df)
    #print(df[["RMSE@Test", "IQRa"]])

    print("\n\nSignificance Test Result:")
    for result in testResults:
        print(result)

    print("\n\n")
    for i, k in enumerate(row_desc):
        rowstring = "\multicolumn{1}{ |c|}{"
        if isinstance(k, str):
            rowstring = rowstring + "\small{" + k + "}}"
        else:
            rowstring = rowstring + "{:0.1f}".format(k) + "}"

        for j, frames in enumerate(dframes):
            if 'R2' in columnsToTest[0]:
                bestIdx = frames[columnsToTest[0]].idxmax(axis=1)
            else:
                bestIdx = frames[columnsToTest[0]].idxmin(axis=1)
                
            toAdd = "{:0.2e}".format(frames[columnsToTest[0]][i])
            if bestIdx == i:
                toAdd = "\\textbf{" + toAdd + "}" + "\hphantom{ *}"
            elif testResults[j][columnsToTest[0]][i-1 if i > bestIdx else i] < 1e-02:
                toAdd =  toAdd + " *"
            else:
                toAdd =  toAdd + "\hphantom{ *}"
            rowstring = rowstring + " & " + toAdd

            toAdd = "{:<0.2e}".format(frames["IQRa"][i])
            if bestIdx == i:
                toAdd = "\\textbf{" + toAdd + "}"
            rowstring = rowstring + " & " + toAdd

        rowstring = rowstring + "\\\\"
        print(rowstring)

def binomfortour(k=11):
    toursize_max = 60
    toursize = np.arange(1, toursize_max,1)
    #print(toursize)
    popsize = 800
    probabilities = toursize/popsize
    #print(probabilities)
    test = binom.pmf(k=k, n=popsize, p=probabilities)          #P(X  = k)
    test1 = binom.cdf(k=k, n=popsize, p=probabilities)         #P(X <= k)
    test2 = 1 - binom.cdf(k=k-1, n=popsize, p=probabilities)   #P(X >= k)                       
    
    #print(np.round(test, decimals=2))
    #print(np.round(test1, decimals=2))
    #print(np.round(test2, decimals=2))

    plt.bar(x=toursize, height=test1, width=0.6)
    xticks = np.arange(5, toursize_max, 5)
    plt.xticks(xticks, xticks)
    plt.title("P(X >= "+str(k)+")")
    plt.show()


def plot_funcs():
    url = 'data/f1_n2000_python.csv'
    df = pd.read_csv(url, header=None)
    df.columns = ["x0","x1","x2","x3","y"]
    #df.columns = ["x0","x1","y"]
    sns.set(style="whitegrid")
    #sns.pairplot(df)
    print(df.mean())
    sns.distplot(df['y'])
    plt.savefig("plotting/plots/func1_bins.PNG", dpi=300)
    plt.show()

def plotErrorHistogram():
    dataset="f1_n2000_python"
    my_data = np.genfromtxt("data/" + dataset + ".csv", delimiter=',')
    X = my_data[:,:-1]
    y = my_data[:,-1]
    y_pred = func1(X[:,0], X[:,1], X[:,2], X[:,3]) # - func2(U, V, .5, .5)
    y_error = abs(y - y_pred)
    sns.set(style="white")

    sns.kdeplot(y, y_error, n_levels=20)
    plt.savefig("plotting/plots/func1b.PNG", dpi=500)
    plt.show()


def plotlengthscross():
    sns.set(style="whitegrid")
    url = '/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/cross/live4/' + 'f1_lengths.csv'
    df = pd.read_csv(url)
    """plt.plot(df['Cross'],df.loc[:, df.columns != 'Cross'], marker="v","o","s","x")
    plt.legend([df.columns[1],df.columns[2],df.columns[3],df.columns[4]], ncol=2, loc='upper left')
    """
    ax = df.plot(x='Crossover', kind='line', style=['-', '--', '-.', '.-'], legend=True, xticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax.set_xlabel("P(Crossover)")
    ax.set_ylabel("Length")
    fig = ax.get_figure()
    plt.savefig("plotting/plots/crosslengths_f1.PNG", dpi=500)
    plt.show()

def boxplotviolionplot(filecompare, plotname, boxplot=False):
    diruri = "/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/algorithms/first/summed/csv/"
    columnnametojoin = "RMSE@Test"

    #read file names and cull by their name
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(diruri) if isfile(join(diruri, f)) ]
    onlyfiles.sort()
    finalfiles = []
    for p in onlyfiles:
        for k in filecompare:
            if k in p:
                finalfiles.append(p)
    

    dframes = []
    for p in finalfiles:
        print(p)
        df = pd.read_csv(diruri + p, header=None)
        df.columns = ["R2@TRain","R2@Test","RMSE@Train","RMSE@Test","Length", "Kommenda"]
        dframes.append(df[columnnametojoin])
    df = pd.concat(dframes, axis=1)
    df.columns = ["EplexGP","St.GP","PGP Kom.","PGP Len."]

    sns.set(style="whitegrid")
    if boxplot:
        sns.boxplot(data=df, palette="Set3")
    else:
        sns.violinplot(data=df, palette="Set3")
    sns.despine(left=True)
    plt.ylabel(columnnametojoin)
    plt.savefig("plotting/plots/" + plotname + ".PNG", dpi=500)
    plt.show()


if __name__ == "__main__":
    #plotError()
    #y_bins()
    #plot_paretofronts()
    #plotFunc3()
    #plotFunc2()
    #boxplots("tournament_length_scores.csv")
    #boxplots("tournament_kommenda_rmsetrain.csv")
    """significancetest("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/parsimony/drittes/live/summed/csv/", 
                        2, 
                        15,
                        [7e-06, 7e-05, 4e-06, 4e-05, 1e-06, 1e-05, 7e-02, 4e-02, 1e-02, 7e-03, 4e-03, 1e-03, 7e-04, 4e-04, 1e-04])"""
    """significancetest("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/cross/live4/summed/csv/", 
                       8, 
                       11,
                       [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])"""

    """significancetest("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/algorithms/first/summed/csv/", 
                       6, 
                       4,
                       [1,2,3,4])"""
    """signigicancetestpandas("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/parsimony/drittes/live/summed/csv/", 
                        2, 
                        15,
                        [1e-06, 4e-06, 7e-06, 1e-05, 4e-05, 7e-05, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07],
                        ["RMSE@Test","Length"],
                        reverse=False)"""

    """signigicancetestpandas("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/cross/live4/summed/csv/", 
                       4,
                       11,
                       [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                       ["RMSE@Test","Length"],
                       reverse=False)"""

    """signigicancetestpandas("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/cross/live4/summed/csv/", 
                       4,
                       2,
                       [1.0, 0.0],
                       ["RMSE@Test","Length"],
                       reverse=False)"""
    """signigicancetestpandas2("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/algorithms/first/summed/csv/", 
                       2, 
                       4,
                       ["EplexGP","StandardGP","PGP Kom.","PGP Length"],
                       ["R2@Train","R2@Test","Length"],
                       reverse=True)"""
    """for k in range(1,30,1):
        binomfortour(k = k)"""

    """signigicancetestpandas3("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/other_regr/csv/", 
                       2, 
                       7,
                       ["MLP","EplexGP","PGP Kommenda","GaussProcess","GradientTrees","Linear","Polynomial"],
                       ["RMSE@Test","RMSE@Train"],
                       reverse=True)"""
    #plot_gpmodeldata("/mnt/daten/dokumente/nextcloud/Bachelorarbeit/results/test gpmodels/crossruns/gpmodels/")
    #plot_funcs()
    #plotlengthscross()
    #boxplotviolionplot(["f2_alt"], "boxplot_f2", boxplot=True)
    #boxplotviolionplot(["f1_n2000"], "boxplot_f1", boxplot=True)
    #plot_csvruns()
    plotErrorHistogram()
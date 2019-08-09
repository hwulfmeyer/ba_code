
import argparse
import datetime
import itertools
import math
import pickle
import sys
import time
from copy import deepcopy
import gplearn.fitness
import numpy as np
import sympy
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import gputils

gens_total = 1250*10
crossvalidations = 1

def gpstart(est_gp, fileprefix, X_train, X_test, y_train, y_test):
    gplogger_est = gputils.Mylogger(folder="runs/live/crossruns", fileprefix=fileprefix, filesuffix="est")
    gplogger_est.start()
    est_gp_pipeline = make_pipeline(PolynomialFeatures(degree=3), est_gp)
    # run genetic programming regression
    internal_gens = 25
    gens_now = internal_gens
    est_gp_pipeline.steps[-1][1].set_params(generations=gens_now)
    starttime = time.time() 
    est_gp_pipeline.fit(X_train, y_train)
    # every 'internal_gens' generations put out the current best program
    while gens_now <= gens_total-1:
        gens_now = gens_now + internal_gens
        gplogger_est.stop()
        gplogger_est.start()
        est_gp_pipeline.steps[-1][1].set_params(generations=gens_now, warm_start=True)
        est_gp_pipeline.fit(X_train, y_train)

    endtime = time.time()
    timediff = endtime - starttime
    timediff_h = math.floor(timediff/3600)
    timediff_min = math.floor(timediff/60 - timediff_h*60)
    timediff_sec = math.floor(timediff - (timediff_h*60 + timediff_min)*60)

    est_gp = est_gp_pipeline.steps[-1][1]    # extract est_gp from pipeline

    print("\n"*4)
    print("Time: {}h:{}m:{}s".format(timediff_h, timediff_min, timediff_sec))
    print("####   Data for the best program   ####")
    best_gp = est_gp._program
    print("Best Program: " + str(best_gp))
    infix = best_gp.InfixExpression()
    print("Best Program Infix: " + str(infix))
    print("Best Program Complexity: " + str(best_gp.complexity_))
    print("Raw Fitness: " + str(best_gp.raw_fitness_))
    print("Penalized Fitness: " + str(best_gp.fitness_))
    gputils.est_evaluation(est_gp_pipeline, X_train, X_test, y_train, y_test)
    print()
    print(est_gp.get_params())
    gplogger_est.stop()
    print(" {} is DONE ".format(fileprefix))
    return est_gp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepr', dest='fileprefix', type=str)
    parser.add_argument('--pgp', dest='paretogp', action='store_true')
    parser.add_argument('--no-pgp', dest='paretogp', action='store_false')
    #parser.add_argument('--pgp_l', dest='paretogp_lengths',  nargs='+', type=int)
    parser.add_argument('--cmplx', dest='complexity', type=str)
    parser.add_argument('--sel', dest='selection', type=str)
    #parser.add_argument('--to_size', dest='tournament_size', type=int)
    parser.add_argument('--el_size', dest='elitism_size', type=int)
    #parser.add_argument('--par_coe', dest='parsimony_coefficient', type=float)
    #parser.add_argument('--p_cr', dest='p_crossover', type=float)
    #parser.add_argument('--p_subm', dest='p_subtree_mutation', type=float)
    #parser.add_argument('--p_pm', dest='p_point_mutation', type=float)
    #parser.add_argument('--p_pr', dest='p_point_replace', type=float)
    parser.add_argument('--n_jobs', dest='n_jobs', type=int)

    if len(sys.argv) > 1:                                 
        args = vars(parser.parse_args())
        args1 = dict(itertools.islice(args.items(), 1, None))
        fileprefix = args["fileprefix"]
        pgp = args["paretogp"]
    else:
        pgp = False
        fileprefix = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    
    length_coefficients = None #[7e-2, 4e-2, 1e-2, 7e-3, 4e-3, 1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 4e-5, 1e-5, 7e-6, 4e-6, 1e-6]
    
    
    data_sets = ["f1_n2000_python", "f2_alt_n2000_python"] #"f2_noisy_1p_n2000_python", "f2_noisy_5p_n2000_python", "f1_n2000_python", "f1_noisy_1p_n2000_python", "f1_noisy_5p_n2000_python"]
    tournament_sizes = [11]
    p_crossover = 0.5
    random_state = np.random.RandomState(6619927)
    seeds = random_state.randint(np.iinfo(np.int32).max, size=crossvalidations)
    for dataset in data_sets:
        if "f2" in dataset:
            f_set = ('add', 'sub', 'div', 'mul', 'sin', 'cos')
            length_coefficients = 1e-03
            if 'eplex' in fileprefix:
                seeds = [816473979]
            if 'stgp' in fileprefix:
                seeds = [797446139]
            if 'pgp_kom' in fileprefix:
                seeds = [1682807055]
            if 'pgp_len' in fileprefix:
                seeds = [44916018]
        elif "f1" in dataset:
            f_set = ('add', 'sub', 'div', 'mul')
            length_coefficients = 7e-05
            if 'eplex' in fileprefix:
                seeds = [26519714]
            if 'stgp' in fileprefix:
                seeds = [797446139]
            if 'pgp_kom' in fileprefix:
                seeds = [981682268]
            if 'pgp_len' in fileprefix:
                seeds = [1449396245]

        my_data = np.genfromtxt("data/" + dataset + ".csv", delimiter=',')
        X = my_data[:,:-1]
        y = my_data[:,-1]
        test_size = 0.5    #1 - (400.0/X.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        gplogger_run = gputils.Mylogger(folder="runs/live", fileprefix=dataset+"__"+fileprefix, filesuffix="eval")
        gplogger_run.start()
        print('{:<11} {:<14} {:<11} {:<14} {:<15} {:<18} {:<15} {:<18} {:<11} {:<14} {:<15} {:<18} {:<30}'.format('~|R2@Train','IQR|R2@Train', '~|R2@Test','IQR|R2@Test', '~|RMSE@Train', 'IQR|RMSE@Train', '~|RMSE@Test', 'IQR|RMSE@Test', '~|Length', 'IQR|Length', '~|Kommenda', 'IQR|Kommenda', 'Config'))
        gplogger_run.stop()
        
        for toursize in tournament_sizes:           # entry point for loops over parameters
            p_mutations = (1.0 - p_crossover) / 2.0
            rmses_train = []
            rmses_test = []
            scores_train = []
            scores_test = []
            lengths = []
            kommendas = []
            #filemiddle = "__pc"+str(p_crossover)
            #filemiddle = "__pcoe"+str(length_coefficient)
            filemiddle = ""
            filemiddle = "tour" + str(toursize)
            gplogger_cv_sum = gputils.Mylogger(folder="runs/live/summed", fileprefix=dataset+"__"+fileprefix+filemiddle, filesuffix="eval")
            gplogger_cv_sum.start()
            print('{:<10} {:<10} {:<14} {:<14} {:<10} {:<14}'.format('R2@Train','R2@Test', 'RMSE@Train', 'RMSE@Test', 'Length', 'Kommenda'))
            gplogger_cv_sum.stop()

            gplogger_cv_sum_csv = gputils.Mylogger(folder="runs/live/summed/csv", fileprefix=dataset+"__"+fileprefix+filemiddle, filesuffix="eval")
            for i, seed in enumerate(seeds):
                est_gp = SymbolicRegressor( population_size=800,
                                    stopping_criteria = 1e-10,
                                    const_range = (.0000000001,2.0),
                                    init_depth = (3,6),
                                    init_method = 'half and half',
                                    function_set = f_set,
                                    metric = 'rmse',
                                    max_samples = 1.0,
                                    low_memory = True,
                                    n_jobs = 4,
                                    verbose = 1,
                                    random_state = seed,
                                    p_hoist_mutation=0.0)

                est_gp.set_params(paretogp_lengths = (5,250),
                            paretogp = False,
                            complexity = 'length',
                            selection = 'tournament',
                            elitism_size = 1,
                            tournament_size = toursize,
                            parsimony_coefficient = length_coefficients, # = 0.0,
                            p_crossover = 0.1,#p_crossover,
                            p_subtree_mutation = 0.5, #p_mutations,
                            p_point_mutation = 0.3, #p_mutations,
                            p_point_replace = 0.05,
                            p_gs_crossover = 0.05,
                            p_gs_mutation = 0.05,
                            gs_mutationstep = 0.001
                            )

                if pgp or est_gp.get_params()['paretogp']:
                    est_gp.set_params(parsimony_coefficient = 0.0)

                if len(sys.argv) > 3:      
                    est_gp.set_params(**args1)       
                
                est = None
                filename = dataset+"__"+fileprefix+filemiddle+"____"+str(i)
                try:
                    est = gpstart(est_gp, filename, X_train, X_test, y_train, y_test)

                    rmse = math.sqrt(mean_squared_error(est.predict(X_test), y_test))
                    rmses_test.append(rmse)
                    rmses_train.append(est._program.raw_fitness_)
                    scores_train.append(est.score(X_train, y_train))
                    scores_test.append(est.score(X_test, y_test))
                    lengths.append(est._program.length_)
                    kommendas.append(est._program.kommenda_)
                    gplogger_cv_sum.start()
                    print('{:<10.3f} {:<10.3f} {:<14.3e} {:<14.3e} {:<10d} {:<14.3e}'.format(scores_train[-1], scores_test[-1], rmses_train[-1], rmses_test[-1], lengths[-1], kommendas[-1]))
                    gplogger_cv_sum.stop()
                    gplogger_cv_sum_csv.start()
                    print('{},{},{},{},{},{}'.format(scores_train[-1], scores_test[-1], rmses_train[-1], rmses_test[-1], lengths[-1], kommendas[-1]))
                    gplogger_cv_sum_csv.stop()
                except Exception as e:
                    print(str(e))
                    gplogger_cv_sum.start()
                    print('{},{},{},{},{},{}'.format(0,0,0,0,0,0))
                    gplogger_cv_sum.stop()
                    gplogger_cv_sum_csv.start()
                    print('{},{},{},{},{},{}'.format(0,0,0,0,0,0))
                    gplogger_cv_sum_csv.stop()
                    pass
                """
                try:
                    gputils.dumpmodel(est, filename)
                except:
                    print("Could not save model of " + filename)
                    pass
                #gputils.createimage(est, dataset+"__"+fileprefix+filemiddle+"_"+str(i))"""
            gplogger_run.start()
            print('{:<11.3f} {:<14.5f} {:<11.3f} {:<14.5f} {:<15.3e} {:<18.3e} {:<15.3e} {:<18.3e} {:<11.2f} {:<14.2f} {:<15.3e} {:<18.3e} {:<30}'.format(np.median(scores_train), np.subtract(*np.percentile(scores_train, [75, 25])), np.median(scores_test), np.subtract(*np.percentile(scores_test, [75, 25])), np.median(rmses_train), np.subtract(*np.percentile(rmses_train, [75, 25])),np.median(rmses_test), np.subtract(*np.percentile(rmses_test, [75, 25])), np.median(lengths), np.subtract(*np.percentile(lengths, [75, 25])), np.median(kommendas), np.subtract(*np.percentile(kommendas, [75, 25])), fileprefix+" <> "+filemiddle))
            gplogger_run.stop()
import datetime
import math
import pickle
import sys
import pydotplus
import numpy as np
import sympy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import parser
import ast


class Mylogger:
    def __init__(self, fileprefix, filesuffix, folder="runs"):
        # log file writing stuff
        self.filename = fileprefix + "_" + filesuffix + ".log"
        self.orig_stdout = sys.stdout
        self.folder = folder
        print("Writing Output to file at:  " + self.filename)
    
    def start(self):
        self.filestream = open(self.folder + "/" + self.filename, 'a')
        sys.stdout = self.filestream
    
    def stop(self):
        self.filestream.close()
        sys.stdout = self.orig_stdout
    
    def stopprintstart(self, output):
        self.stop()
        print(output)
        self.start()


def rmse(y_pred, y):
    return np.sqrt(mean_squared_error(y, y_pred))

def mpe(y_pred, y):
    # mean percentage error
    return np.mean(np.divide(np.abs(y_pred - y), np.abs(y)))


def est_evaluation(estimator, X_train, X_test, y_train, y_test):
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    score_gp = estimator.score(X_test, y_test)
    rmse_gp_train = rmse(estimator.predict(X_train), y_train)
    rmse_gp_test = rmse(estimator.predict(X_test), y_test)
    print("Estimator\t"+ "\tTrain-rmse: " + str(rmse_gp_train) + "\tTest-rmse: " + str(rmse_gp_test)  + "\tTest-R^2: " + str(score_gp))
    return score_gp, rmse_gp_train, rmse_gp_test


# saving gp model as pickle file
def dumpmodel(estimator_gp, name=None):
    if name is None:
        name = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    with open("runs/live/crossruns/gpmodels/" + name + "_gp_model" + ".pkl", 'wb') as f:
        pickle.dump(estimator_gp, f)


# create image of program & save to file
def createimage(estimator_gp, name=None):
    if name is None:
        name = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    graph = estimator_gp._program.export_graphviz()
    graph = pydotplus.graphviz.graph_from_dot_data(graph)
    est_gp_image = graph.create_png()
    with open("runs/test/" + name + "_gp_model" + ".png", 'wb') as f:
        f.write(est_gp_image)


def add(a, b):
    return "(" + str(a) +"+"+ str(b) + ")"

def sub(a, b):
    return "(" + str(a) +"-"+ str(b) + ")"

def div(a, b):
    return "(" + str(a) +"/"+ str(b) + ")"

def mul(a, b):
    return "(" + str(a) +"*"+ str(b) + ")"

def sin(a):
    return "sin(" + str(a) + ")"

def cos(a):
    return "cos(" + str(a) + ")"

def sqrt(a):
    return "cos(" + str(a) + ")"

def parse(eq):
    X0 = "a"
    X1 = "b"
    X2 = "c"
    X3 = "d"
    return eval(eq)


def runfunc(a,b,c,d):
    code = "0.0336*a*b*c - 0.1831*a*b - 0.0592*a*c + 0.1938*a - 0.0825*b*c + 0.2908*b - 0.0560*c**2 - 0.0112*c*d - 0.1583*c - 0.0224*d**3 - 0.0560*d**2 + 0.0780*d + 0.5844"
    #code = parse(code)
    return eval(code)

class InfToPre(ast.NodeVisitor):
    # thanks to https://stackoverflow.com/questions/42590512/how-to-convert-from-infix-to-postfix-prefix-using-ast-python-module
    def __init__(self, expr):
        self.prefix = []
        self.astInfix = ast.parse(str(expr))

    def transform(self):
        self.visit(self.astInfix)

    def f_continue(self, node):
        super(InfToPre, self).generic_visit(node)

    def visit_Add(self, node):
        self.prefix.append('Add')
        self.f_continue(node)

    def visit_Div(self, node):
        self.prefix.append('Div')
        self.f_continue(node)

    def visit_Mult(self, node):
        self.prefix.append('Mul')
        self.f_continue(node)

    def visit_Sub(self, node):
        self.prefix.append('Sub')
        self.f_continue(node)

    def visit_Name(self, node):
        if node.id == 'sin':
            self.prefix.append('Sin')
        elif node.id == 'cos':
            self.prefix.append('Cos')
        else:
            self.prefix.append(node.id)
        self.f_continue(node)

    def visit_Num(self, node):
        self.prefix.append(node.n)
        self.f_continue(node)

    def visit_BinOp(self, node):
        self.visit(node.op)
        self.visit(node.left)
        self.visit(node.right)

    def visit_Call(self, node):
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)

    def visit_Expr(self, node):
        self.f_continue(node)



def func_evaluation(X, y):
    y_pred = runfunc(X[:,0], X[:,1], X[:,2], X[:,3])
    for y_pred_val in y_pred:
        if y_pred_val in (float('NaN'), float('Inf'), -float('Inf')):
            print("true")
    rmse_val = rmse(y_pred, y)
    print("RMSE: " + str(rmse_val))

if __name__ == "__main__":
    dataset="f1_n2000_python"
    my_data = np.genfromtxt("data/" + dataset + ".csv", delimiter=',')
    X = my_data[:,:-1]
    y = my_data[:,-1]
    func_evaluation(X, y)
    
    code = "add(sub(0.926, 1.253), add(0.872, mul(sub(add(X1, add(sub(0.923, 1.253), add(0.872, mul(sub(sub(add(sub(add(add(X0, mul(sub(add(sub(add(sub(1.651, X1), mul(sub(sub(0.923, X3), X3), sub(add(X3, X3), 1.172))), mul(X3, X3)), add(sub(mul(sub(0.923, mul(X3, X3)), add(X3, X3)), X0), mul(sub(X2, 0.923), sub(sub(sub(0.923, sub(X1, sub(sub(sub(0.923, sub(mul(sub(X0, 0.453), sub(sub(0.923, add(X1, X1)), sub(X1, 0.973))), sub(sub(0.926, sub(X2, sub(sub(sub(sub(sub(1.464, X1), add(X1, X1)), X1), X3), X2))), X2))), X2), 0.195))), X2), X1)))), X2), 0.295)), mul(sub(X0, 0.453), sub(sub(sub(1.904, add(X2, X1)), add(X1, X1)), sub(X1, 0.973)))), sub(X2, 0.923)), sub(sub(X2, X2), 0.195)), X2), sub(X1, X0)), 0.195)))), X2), 0.195)))"
    code = parse(code)
    code = "(0.5 + 0.2*a + 0.3*b - 0.2*(a*b))*(1 - 0.3*c - 0.1*c**2) + 0.1*(1 - (d - 0.5)**2)"
    simpli = sympy.simplify(code)
    print(simpli)
    #expre = InfToPre(simpli)
    #expre.transform()
    #print(expre.prefix)
    #print()
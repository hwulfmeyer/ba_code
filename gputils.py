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
    X0 = "X0"
    X1 = "X1"
    X2 = "X2"
    X3 = "X3"
    return eval(eq)


def runfunc(a,b,c,d):
    code = "sub(0.657, mul(0.252, add(mul(0.252, sub(X1, X0)), add(mul(0.168, sub(mul(mul(add(div(X0, 0.347), mul(0.168, sub(add(X2, add(div(sub(sub(mul(0.679, add(X3, mul(1.409, mul(1.409, sub(X2, X0))))), X3), X3), X1), 0.698)), X3))), X1), X2), X0)), add(mul(0.168, mul(mul(0.679, add(X3, mul(1.409, mul(1.409, sub(X2, X0))))), sub(X2, 0.828))), add(mul(0.168, mul(X3, sub(mul(X3, sub(X0, sub(0.556, X3))), X0))), add(mul(sub(0.836, X2), mul(sub(0.657, X1), sub(0.556, X0))), sub(X2, X1))))))))"
    code = parse(code)
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



def func_evaluation(X_train, X_test, y_train, y_test):
    rmse_gp_train = rmse(runfunc(X_train[:,0], X_train[:,1], X_train[:,2], X_train[:,3]), y_train)
    rmse_gp_test = rmse(runfunc(X_test[:,0], X_test[:,1], X_test[:,2], X_test[:,3]), y_test)
    print("Train-rmse: " + str(rmse_gp_train) + "\t\tTest-rmse: " + str(rmse_gp_test))

if __name__ == "__main__":
    """train_file="f1_n200_matlab.csv"
    test_file="f1_n1000_python.csv"
    my_data = np.genfromtxt("data/" + train_file, delimiter=',')
    X_train = my_data[:,[0,1,2,3]]
    y_train = my_data[:,4]

    my_data = np.genfromtxt("data/" + test_file, delimiter=',')
    X_test = my_data[:,[0,1,2,3]]
    y_test = my_data[:,4]
    func_evaluation(X_train,X_test, y_train,y_test)
    """
    code = "add(add(add(add(add(add(add(add(add(add(add(add(0.655, mul(0.270, sub(X1, X2))), mul(0.036, sub(mul(add(X0, sub(X0, 0.212)), add(sub(1.029, X1), sub(1.033, X1))), X1))), mul(0.001, mul(1.937, add(X3, mul(sub(1.653, mul(mul(mul(add(sub(add(sub(X1, X3), add(X2, X2)), sub(X3, X0)), mul(add(mul(X3, 0.455), X3), X3)), mul(add(X1, X1), X2)), 1.659), 1.659)), add(X2, X2)))))), mul(0.001, sub(mul(mul(div(X3, 0.223), sub(0.645, X3)), add(add(div(1.885, 0.270), div(X3, 0.223)), div(sub(1.667, div(sub(1.659, X1), div(X3, 0.162))), 0.223))), div(sub(sub(X1, sub(sub(X2, X3), 0.212)), sub(X0, 1.390)), div(mul(add(X0, 0.628), add(X2, X1)), 1.756))))), mul(0.001, X0)), mul(0.001, div(X3, 0.223))), mul(0.001, sub(sub(div(X3, 0.223), add(sub(mul(mul(1.995, 0.655), sub(0.645, X3)), X3), div(sub(sub(1.033, X1), sub(X0, 1.390)), add(X3, X2)))), 0.036))), mul(0.001, div(X3, 0.196))), mul(0.001, sub(div(sub(sub(X0, div(X1, 0.426)), sub(add(0.655, mul(mul(1.995, mul(1.995, sub(1.079, X1))), sub(X1, X2))), X0)), add(X1, X2)), sub(1.079, X1)))), mul(0.001, sub(mul(X3, add(1.544, sub(1.659, mul(mul(mul(mul(mul(add(mul(add(X0, 0.744), mul(add(X0, sub(X0, 0.212)), mul(1.995, sub(1.079, X1)))), X3), div(X2, X3)), 1.659), X2), X2), add(X2, X2))))), 0.280))), sub(X0, X0)), mul(0.001, 0.668))"
    simpli = sympy.simplify(parse(code))
    print(simpli)
    expre = InfToPre(simpli)
    expre.transform()
    print(expre.prefix)
    print()
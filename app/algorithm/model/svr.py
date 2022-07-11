#Import required libraries
from math import gamma
from random import Random
import numpy as np, pandas as pd
import joblib
import sys 
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


model_fname = "model.save"
MODEL_NAME = "SupportVectorRegressor"


class SupportVectorRegressor(): 
    
    def __init__(self, C=1.0, kernel="rbf", degree=1, tol=1e-3, gamma="auto") -> None:
        
        self.C= np.float(C)
        self.kernel= kernel
        self.degree= int(degree)
        self.tol= np.float(tol)
        self.gamma= gamma
        self.verbose = False

        self.model = self.build_model()
        
        
        
    def build_model(self): 
        model = SVR(C= self.C, kernel= self.kernel, degree= self.degree, 
                    tol= self.tol, gamma= self.gamma, verbose=self.verbose)
        return model
    
    
    def fit(self, train_X, train_y):        
                 
    
        self.model.fit(
                X = train_X,
                y = train_y
            )
    
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            # return self.model.score(x_test, y_test)        
            preds = self.model.predict(x_test)
            mse = mean_squared_error(y_test, preds, squared=False)
            return mse
            

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path): 
        svr = joblib.load(os.path.join(model_path, model_fname))
        return svr


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = SupportVectorRegressor.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model



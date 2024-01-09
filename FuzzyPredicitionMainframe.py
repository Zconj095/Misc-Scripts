import fuzzylite as fl
import numpy as np
from sklearn.neural_network import MLPRegressor

class FuzzyPredictionMainframe:
    def __init__(self):
        self.fuzzy_engine = fl.Engine()
        self.mlp = MLPRegressor(hidden_layer_sizes=(100,50,25))
        
    def train(self, X, y):
        self.mlp.fit(X, y)
        self.define_fuzzy_sets(X, y)
        
    def define_fuzzy_sets(self, X, y):
        # Analyze data ranges to define fuzzy membership functions
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        
        mf_low = fl.TrapezoidalMF(x_min[0], x_min[0], x_min[0]+0.1*(x_max[0]-x_min[0]), x_min[0]+0.25*(x_max[0]-x_min[0]))
        mf_medium = fl.TriangularMF(x_min[0]+0.25*(x_max[0]-x_min[0]), 0.5*(x_min[0]+x_max[0]), x_max[0]-0.25*(x_max[0]-x_min[0]))  
        mf_high = fl.TrapezoidalMF(x_max[0]-0.25*(x_max[0]-x_min[0]), x_max[0]-0.1*(x_max[0]-x_min[0]), x_max[0], x_max[0])
        
        self.fuzzy_engine.add_input_variable("X")
        self.fuzzy_engine.add_output_variable("Y")
        
        self.fuzzy_engine.add_membership_functions([mf_low, mf_medium, mf_high])
        
    def predict(self, X):
        mlp_pred = self.mlp.predict(X)
        
        fuzzified_input = self.fuzzy_engine.fuzzify("X", X) 
        inferred_output = self.fuzzy_engine.infer([("X", fuzzified_input)])
        
        defuzzified = self.fuzzy_engine.defuzzify("Y", inferred_output, 'centroid')
        
        # Combine MLP and fuzzy outputs 
        y_pred = (mlp_pred + defuzzified) / 2.0  
        
        return y_pred

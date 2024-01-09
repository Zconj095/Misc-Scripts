import numpy as np
import fuzzylite as fl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class FuzzyDeepNetwork:
    def __init__(self):
        self.fuzzy_engine = fl.Engine()
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=75, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model
        
    def define_fuzzy_sets(self, X):
        x_min = np.min(X, axis=0) 
        x_max = np.max(X, axis=0)
        
        # Define fuzzy membership functions
        mf_low = fl.TrapezoidalMF(x_min, x_min, x_min+0.1*(x_max-x_min), x_min+0.3*(x_max-x_min))
        mf_medium = fl.TriangularMF(x_min+0.3*(x_max-x_min), 0.5*(x_min+x_max), x_max-0.3*(x_max-x_min))
        mf_high = fl.TrapezoidalMF(x_max-0.3*(x_max-x_min), x_max-0.1*(x_max-x_min), x_max, x_max)

        self.fuzzy_engine.add_input_variable("input")
        self.fuzzy_engine.add_output_variable("output")
        self.fuzzy_engine.add_membership_functions([mf_low, mf_medium, mf_high])
        
    def train(self, X, y):  
        self.define_fuzzy_sets(X) 
        fuzzified_X = self.fuzzy_engine.fuzzify("input", X)
        
        self.model.fit(fuzzified_X, y, epochs=150, verbose=0)  
        
    def predict(self, X):
        fuzzified_X = self.fuzzy_engine.fuzzify("input", X)
        model_pred = self.model.predict(fuzzified_X)
        
        return model_pred

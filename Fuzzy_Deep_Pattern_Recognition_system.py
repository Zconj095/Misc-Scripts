import numpy as np
import fuzzylite as fl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class FuzzyDeepPatternRecognizer:
    def __init__(self):
        self.fuzzy_engine = fl.Engine()
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(30,)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def define_fuzzy_sets(self, X):
        # Define fuzzy membership functions
        min_val = min(X)
        max_val = max(X)
        
        # Triangular, trapezoidal, bell MFs     
        low = fl.TriangularMF(min_val, min_val*1.25, X.mean()*0.75)
        mid = fl.TrapezoidalMF(X.mean()-X.std(), X.mean()-X.std()*0.5, X.mean()+X.std()*0.5, X.mean()+X.std())  
        high = fl.BellMF(X.mean(), X.std()*3, max_val)
               
        self.fuzzy_engine.add_input_variable("input")
        self.fuzzy_engine.add_output_variable("output")
        self.fuzzy_engine.add_membership_functions([low, mid, high])

    def preprocess(self, X):   
        fuzzified_X = self.fuzzy_engine.fuzzify("input", X)  
        return fuzzified_X
    
    def fit(self, X, y): 
        self.define_fuzzy_sets(X)  
        fuzzified_X = self.preprocess(X)
        self.model.fit(fuzzified_X, y)

    def predict(self, X):
        fuzzified_X = self.preprocess(X)
        return self.model.predict(fuzzified_X)

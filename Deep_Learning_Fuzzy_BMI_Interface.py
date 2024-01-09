import numpy as np
import fuzzylite as fl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class FuzzyDLBMI:
    def __init__(self):
        self.fuzzy_engine = fl.Engine()  
        self.model = self.build_model()
        self.encoder = SelfAttentionEncoder()
        
    def build_model(self):
        model = Sequential() 
        model.add(Dense(64, activation='relu', input_shape=(50,))) 
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        
        return model
        
    def define_fuzzy_sets(self, brain_signals):
        # Define fuzzy sets 
        mf_low = fl.TrapezoidalMF(min(brain_signals), min(brain_signals), min(brain_signals)*1.1, min(brain_signals)*1.25)
        # Other MFs...
        
        self.fuzzy_engine.add_input_variable("brain_signal")
        self.fuzzy_engine.add_output_variable("computer_action")  
        self.fuzzy_engine.add_membership_functions([mf_low...])
        
    def train(self, X, y):
        self.define_fuzzy_sets(X)
        fuzzified_X = self.fuzzy_engine.fuzzify("brain_signal", X)
        
        # Encode string labels 
        y_cat = self.encoder.encode_output(y)  
        
        # Train model 
        self.model.fit(fuzzified_X, y_cat, epochs=150, verbose=0)
            
    def predict(self, X):
        fuzzified_X = self.fuzzy_engine.fuzzify("brain_signal", X)   
        preds = self.model.predict(fuzzified_X)
        pred_labels = self.encoder.decode_output(preds)  
        return pred_labels

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense

class HyperstateMaingrader:
    def __init__(self, problem):
        self.problem = problem  
        self.hypernet = self.build_hypernet(problem)
    
    def build_hypernet(self, problem):
        # Create RNN to model hyperstate of problem
        input_layer = LSTM(64, return_sequences=True, 
                           input_shape=(None, problem.num_variables))
        
        # Intermediate LSTMs
        lstm_1 = LSTM(32, return_sequences=True)
        
        # Output LSTM 
        output_layer = LSTM(16)
        
        inputs = input_layer(problem.input_data)
        x = lstm_1(inputs)
        outputs = output_layer(x)
        
        model = Model(inputs, outputs)
        return model
    
    def identify_maingrades(self):
        with GradientTape() as tape:
            hyperstate = self.hypernet(self.problem.input_data)
            loss = self.problem.loss_fn(hyperstate)
            
        grads = tape.gradient(loss, self.hypernet.weights) 
        
        # Analyze gradients to identify key factors
        main_grads = []
        for i in range(len(grads)):
            if abs(grads[i]) > 0.75*max(grads):  
                main_grads.append(self.hypernet.weights[i])
                
        return main_grads
    
    def adapt_solution(self, solution):
        maingrades = self.identify_maingrades()
        
        # Evolve solution to focus on maingrades
        for mg in maingrades:
            solution = evolve(solution, mg)
            
        return solution

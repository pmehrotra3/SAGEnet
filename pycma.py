import cppnn
import numpy as np
import cma  # pycma package is imported as "cma"

class CMA_nn_policy:
    
    def __init__(self,input_size, output_size, hidden_layers, environment, verbose=False):
        
        self.nn = cppnn.NeuralNetwork(input_size, output_size, hidden_layers)
        self.optimizer = cma.CMAEvolutionStrategy(self.nn.get_param(), 0.5)  # Initial weights and sigma
        self.environment = environment
        self.verbose = verbose
        
        
    def learn(self, total_timesteps,logger=None):
        
        for gen in range(generations):
            solutions = self.optimizer.ask()
            
            fitness_values = []
            for sol in solutions:
                self.nn.set_param(sol)
                score = fitness_function(self.nn)
                fitness_values.append(score)
                
            self.optimizer.tell(solutions, fitness_values)
                



#print("\nFinal optimized weights:")
#print(policy.optimizer.result.xbest)
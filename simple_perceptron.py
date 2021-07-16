import numpy as np

#activation function definition
def sigmoid(x): #function
    return 1 / (1 + np.exp(-x))

def sigmoid_der(sig): #first derivative (input should be sigmoid returned value!)
    return sig * (1 - sig)

#training input data
input_matrix = [[0,0,1],
                [1,1,1],
                [1,0,1],
                [0,1,1]]
training_input = np.array(input_matrix)

#training output data
output_vector = [0,1,1,0]
training_output = np.array(output_vector).transpose()

#random synapses weights initialization
np.random.seed(1)
synapses_weights = 2 * np.random.rand(3) - 1

print('Initial synapses weights: ')
print(synapses_weights)
print('------------------')

# print(training_input)
# print(training_output)

#learning process
for i in range(10):

    output = sigmoid(np.matmul(training_input,synapses_weights)) #values returned from the activation function

    iteration_err = output_vector - output #error in the current iteration

    adjustment = iteration_err * sigmoid_der(output) #adjustmen values for synapes weights

    synapses_weights += np.matmul(training_input.T, adjustment) #synapses weights update

print('Output:')
print(output)
print('------------------')
print('Error after last iteration:')
print(iteration_err)
print('------------------')

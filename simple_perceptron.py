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
#np.random.seed(1)
synapses_weights = 2 * np.random.rand(3) - 1

print('Initial synapses weights: ')
print(synapses_weights)
print('------------------')

# print(training_input)
# print(training_output)

#initial average error
max_error = [1,1,1,1]
max_error = np.max(np.array(max_error))
my_iterator = 0

while max_error > 0.1:

#learning process
#for i in range(1):

    output = sigmoid(np.matmul(training_input,synapses_weights)) #values returned from the activation function

    iteration_err = output_vector - output #error in the current iteration
    # error = expected output value - actual output

    adjustment = iteration_err * sigmoid_der(output) #adjustmen values for synapes weights
    # adjustment coef = error * sigmoid derivative(x1,x2,x3,w1,w2,w3)

    synapses_weights += np.matmul(training_input.T, adjustment) #synapses weights update
    # xij - i element from j set
    # w1 = w1 + (x11*a1 + x12*a2 + x13*a3 + x14*a4)
    # w2 = w2 + (x21*a1 + x22*a2 + x23*a3 + x24*a4)
    # w3 = w3 + (x31*a1 + x32*a2 + x33*a3 + x34*a4)

    #average error
    max_error = np.max(np.absolute(np.array(iteration_err)))
    my_iterator += 1


print('Output:')
print(output)
print('------------------')
print('Error after last iteration:')
print(iteration_err)
print('------------------')
print('Synapses weights after learning process:')
print(synapses_weights)
print('------------------')
print('Number of iterations = ' + str(my_iterator))


# class for variables in neural network
class Variable:
    def __init__(self, value, prev = (), op = ''):
        self.value = value              # value of variable
        self.grad = 0.0                 # gradient value
        self.prev = set(prev)           # parent values
        self.op = op                    # operation that produced value from parents
        self.backward = lambda: None    # function for computing gradient

    # overload addition operation
    def __add__(self, other):
        # ensure other is an instance of Variable class
        if isinstance(other, Variable):
            other = other
        else:
            other = Variable(other)
        
        # create a new variable and store new value, parents, and operator
        out = Variable(self.value + other.value, (self, other), '+')
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward

        return out
    
    # overload multiplication
    def __mul__(self, other):
        # ensure other is an instance of Variable class
        if isinstance(other, Variable):
            other = other
        else:
            other = Variable(other)
        
        out = Variable(self.value * other.value, (self, other), '*')

        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out.backward = backward

        return out
    
    # overload division
    def __truediv__(self, other):
        # ensure other is an instance of Variable class
        if isinstance(other, Variable):
            other = other
        else:
            other = Variable(other)

        out = Variable(self.value / other.value, (self, other), '/')

        def backward():
            self.grad += (1 / other.value) * out.grad
            other.grad += (-self.value / (other.value ** 2)) * out.grad
        out.backward = backward

        return out
        
    # overload subtraction
    def __sub__(self, other):
        return self + (-other)

    # overload negation
    def __neg__(self):
        return self * -1
    
    # perform backpropogation
    def backprop(self):
        topo = []
        visited = set()

        def build(variable):
            if variable not in visited:
                visited.add(variable)
                for child in variable.prev:
                    build(child)
                topo.append(variable)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node.backward()

# define an exponential function to avoid use of math library
def exp(x, terms = 50):
    result = 1.0
    term = 1.0

    # use taylor expansion
    for n in range(1, terms):
        term *= x / n
        result += term
    return result

# define a function for generating random numbers without random library
class PRNG:
    # use Linear Congruential Generatator
    def __init__(self, seed):
        self.current_seed = seed
        self.multiplier = 16807
        self.increment = 0
        self.modulus = 2147483647
    
    # get next random number using LCG formula
    def next_random(self):
        self.current_seed = (self.multiplier * self.current_seed + self.increment) % self.modulus
        return self.current_seed
    
    # get a random float between 0 and 1
    def get_random_float(self):
        return self.next_random() / self.modulus
    
    # get a random float between number range
    def get_random_float_range(self, min_val, max_val):
        return min_val + (max_val - min_val) * self.get_random_float()
    
    def get_random_int_range(self, min_val, max_val):
        return int(self.get_random_float_range(min_val, max_val))

# initialize pseudo-random number generator
prng = PRNG(seed = 23422)

# sigmoid function
def sigmoid(x):
    if x.value > 0:
        sig_value = 1 / (1 + exp(-x.value))
    else:
        ex = exp(x.value)
        sig_value = ex / (1 + ex)

    out = Variable(sig_value, (x, ), 'sigmoid')

    def backward():
        out_grad = out.grad
        out_value = out.value

        # compute signmoid derivative
        x.grad += out_grad * (out_value * (1 - out_value))
    out.backward = backward

    return out

# class for MLP neuron
class Neuron:
    def __init__(self, n_inputs, layer_num, neuron_num):
        self.layer_num = layer_num
        self.neuron_num = neuron_num
        self.W = [Variable(prng.get_random_float_range(-1,1)) for i in range(n_inputs)]
        self.b = Variable(0.0) 

    # compute activation for neuron
    def __call__(self, x):
        weighted_sum = self.b
        for wi, xi in zip(self.W, x):
            weighted_sum = weighted_sum + wi * xi
        return sigmoid(weighted_sum)

# class for MLP layer  
class Layer:
    def __init__(self, n_inputs, n_outputs, layer_num):
        self.neurons = []
        for i in range(n_outputs):
            neuron = Neuron(n_inputs, layer_num, i)
            self.neurons.append(neuron)
    
    # compute activations for layer
    def __call__(self, x):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(x))
        return outputs

# class for multi-layer perceptron
class MLP:
    def __init__(self, n_inputs, hidden_sizes, n_outputs):
        sizes = [n_inputs] + hidden_sizes + [n_outputs]
        self.layers = []
        for i in range(len(sizes) - 1):
            layer = Layer(sizes[i], sizes[i + 1], i)
            self.layers.append(layer)
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# loss function
def loss_function(y_pred, y_true):
    differences = [(yp - yt) * (yp - yt) for yp, yt in zip(y_pred, y_true)]
    return sum(differences, Variable(0.0))

# xor dataset
def xor_dataset():
    xor = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    return xor

# two bit adder dataset
def two_bit_adder_dataset():
    dataset = []

    for a0 in [0.0, 1.0]:
        for b0 in [0.0, 1.0]:
            for c0 in [0.0, 1.0]:
                for a1 in [0.0, 1.0]:
                    for b1 in [0.0, 1.0]:
                        s0 = (a0 + b0 + c0) % 2
                        carry_0 = (a0 + b0 + c0) // 2

                        s1 = (a1 + b1 + carry_0) % 2
                        carry_1 = (a1 + b1 + carry_0) // 2

                        c2 = carry_1

                        inputs = [a0, b0, c0, a1, b1]
                        outputs = [s0, s1, c2]

                        dataset.append((inputs, outputs))

    return dataset

# function to get prediction on new input
def predict(net, x_raw):
    # convert value to Variable object
    x_vars = [Variable(v) for v in x_raw]
    y_pred_vars = net(x_vars)
    return [v.value for v in y_pred_vars]

def shuffle_list(list):
    # shuffle the dataset
    for _ in range(len(list)):
        # get two random index values
        idx1 = prng.get_random_int_range(0, len(list))
        idx2 = prng.get_random_int_range(0, len(list))

        # swap the elements at the two indicies
        temp = list[idx1]
        list[idx1] = list[idx2]
        list[idx2] = temp
    
    return list

def test_train_split(dataset, percent_train):
    num_train = int(percent_train * len(dataset))

    if percent_train != 1.0:
        dataset = shuffle_list(dataset)
  
    # split into training and testing sets
    training_dataset = dataset[0:num_train]
    testing_dataset = dataset[num_train:]

    return training_dataset, testing_dataset

def evaluate_test_set(net, test_set):
    total_loss = 0.0
    for x_raw, y_raw in test_set:
        x_vars = [Variable(v) for v in x_raw]
        y_vars = [Variable(v) for v in y_raw]
        y_pred = net(x_vars)
        total_loss += loss_function(y_pred, y_vars).value
    return total_loss / len(test_set)

def train_mlp(net, training_set, learning_rate, num_epochs):
    # train MLP
    total_loss = 'inf'
    for epoch in range(num_epochs):
        total_loss = 0.0

        for x_raw, y_raw in training_set:
            # convert dataset values into instances of Variable class
            x_vars = [Variable(v) for v in x_raw]
            y_vars = [Variable(v) for v in y_raw]

            # foward pass
            y_pred = net(x_vars)
            loss = loss_function(y_pred, y_vars)
            
            # reset gradients for weights and biases
            for layer in net.layers:
                for neuron in layer.neurons:
                    for W in neuron.W:
                        W.grad = 0
                    neuron.b.grad = 0
            
            # backprop step
            loss.backprop()

            # update weights and biases
            for layer in net.layers:
                for neuron in layer.neurons:
                    for W in neuron.W:
                        W.value = W.value - (learning_rate * W.grad)
                    neuron.b.value = neuron.b.value - (learning_rate * neuron.b.grad)

            # compute loss
            total_loss += loss.value
        
        # print training progress to terminal
        print(f"Epoch: {epoch}, Loss = {total_loss:.4f}")
    return net, total_loss

# function 
def run_MLP(dataset, 
            dataset_name, 
            percent_train,
            n_inputs, 
            hidden_sizes, 
            n_outputs, 
            learning_rate, 
            num_epochs):
    
    with open(f"{dataset_name}_example.txt", "w") as f:
        # construct MLP
        net = MLP(n_inputs = n_inputs,
                  hidden_sizes = hidden_sizes, 
                  n_outputs = n_outputs)
        
        # train the network
        training_set, testing_set = test_train_split(dataset, percent_train)
        net, total_loss = train_mlp(net, 
                                    training_set = training_set, 
                                    learning_rate = learning_rate, 
                                    num_epochs = num_epochs)
        
        # write nicely formatted output to txt file
        f.write(f'========== MLP OUTPUT FOR {dataset_name} ==========\n\n')
        f.write('Parameters:\n')
        f.write(f'   Number of inputs = {n_inputs}\n')
        f.write(f'   Hidden Layers = {hidden_sizes}\n')
        f.write(f'   Number of outputs = {n_outputs}\n')
        f.write(f'   Learning rate = {learning_rate}\n')
        f.write(f'   Num_epochs = {num_epochs}\n')
        f.write(f'   Train size = {percent_train * 100}%\n')
        f.write('Results:\n')
        f.write(f'   Loss = {total_loss}\n\n')

        f.write('Predictions for all data (rounded to nearest integer):\n\n')
        total = 0.0
        correct = 0.0
        for input in dataset:
            pred_output = predict(net, input[0])
            pred_output = [round(i, 0) for i in pred_output]
            exp_output = input[1]
            exp_output = [round(i, 0) for i in exp_output]

            if pred_output == exp_output:
                correct += 1
            total += 1
            
            f.write(f'Input = {[int(i) for i in input[0]]}, Output = {[int(i) for i in pred_output]}, '
                    f'{'Correct' if pred_output == exp_output else 'Incorrect'}\n')

        accuracy = round(correct / total, 2)
        f.write(f'\nAccuracy = {accuracy * 100}%\n')

        f.write('\n\nTraining set used:\n\n')
        for input in training_set:
            f.write(f'Input = {[int(i) for i in input[0]]}, Output = {[int(i) for i in input[1]]}\n')

        f.write('\nPrecitions for test set used:\n\n')
        if not testing_set:
                f.write('All data used for training and testing\n')
        else:
            for input in testing_set:
                pred_output = predict(net, input[0])
                pred_output = [round(i, 0) for i in pred_output]
                exp_output = input[1]
                exp_output = [round(i, 0) for i in exp_output]

                if pred_output == exp_output:
                    correct += 1
                total += 1
                
                f.write(f'Input = {[int(i) for i in input[0]]}, Output = {[int(i) for i in pred_output]}, '
                        f'{'Correct' if pred_output == exp_output else 'Incorrect'}\n')

    return total_loss

def main():
    '''
    This function calls the function run_MLP
    '''
    # run example on XOR dataset - write results to text file
    run_MLP(dataset = xor_dataset(), 
            dataset_name = 'XOR',
            percent_train = 1.0,
            n_inputs = 2,
            hidden_sizes = [3],
            n_outputs = 1,
            learning_rate = 0.05,
            num_epochs = 20000)
    
    # run example on adder dataset - write results to text file
    run_MLP(dataset = two_bit_adder_dataset(), 
            dataset_name = 'ADDER',
            percent_train = 1.0,
            n_inputs = 5,
            hidden_sizes = [10, 5],
            n_outputs = 3,
            learning_rate = 0.05,
            num_epochs = 1400)
    
    print("\nDone. Check output files for results.\n")

if __name__ == "__main__":
    main()
# class for variables in neural network
class Variable:
    def __init__(self, value, prev = (), op = '', label = ''):
        self.value = value    # value of variable
        self.grad = 0.0       # gradient value
        self.prev = set(prev) # parent values
        self.op = op          # operation that produced value from parents
        self.backward = lambda: None
        self.label = label
    
    def parent_chain(self, wrt):
        chains = []
        seen_chains = set()  # will store chain keys as tuples of variable labels

        def recurse(curr, path):
            if curr == wrt:
                new_chain = path + [wrt]
                # generate a hashable key using labels
                key = tuple(node.label for node in new_chain)
                if key not in seen_chains:
                    chains.append(new_chain)
                    seen_chains.add(key)
                return
            for parent in curr.prev:
                recurse(parent, path + [curr])

        recurse(self, [])
        return chains
    
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

# initialize pseudo-random number generator
prng = PRNG(seed = 23422)

# sigmoid function
def sigmoid(x):
    if x.value > 0:
        sig_value = 1 / (1 + exp(-x.value))
    else:
        ex = exp(x.value)
        sig_value = ex / (1 + ex)

    out = Variable(sig_value, (x, ), 'sigmoid', label = f'sigmoid({x.label})')

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
        self.W = [Variable(prng.get_random_float_range(-1,1), label = f"W_{neuron_num}_{i}_({layer_num})") for i in range(n_inputs)]
        self.b = Variable(0.0, label = f"b_{neuron_num}_({layer_num})") 

    # compute activation for neuron
    def __call__(self, x):
        weighted_sum = self.b
        for wi, xi in zip(self.W, x):
            weighted_sum = weighted_sum + wi * xi
        weighted_sum.label = f'a_{self.neuron_num}_({self.layer_num})'
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
    x_vars = [Variable(v, label = 'x_i') for v, i in enumerate(x_raw)]

    y_pred_vars = net(x_vars)

    return [v.value for v in y_pred_vars]

# function to run xor predictions
def predict_xor(net):
    pred = predict(net, [0, 0])
    print(pred)
    pred = predict(net, [0, 1])
    print(pred)
    pred = predict(net, [1, 0])
    print(pred)
    pred = predict(net, [1, 1])
    print(pred)

# function to run adder predictions
def predict_adder(net):
    pred = predict(net, [0, 0, 0, 0, 0])
    print(pred)
    pred = predict(net, [0, 1, 1, 0, 0])
    print(pred)
    pred = predict(net, [1, 0, 0, 0, 1])
    print(pred)
    pred = predict(net, [1, 1, 0, 0, 1])
    print(pred)
    pred = predict(net, [1, 1, 1, 1, 1])
    print(pred)

def main():
    # create MLP
    net = MLP(n_inputs = 2, hidden_sizes = [3], n_outputs = 1)

    dataset = xor_dataset()
    # dataset = two_bit_adder_dataset()
    
    # train MLP
    last_loss = Variable(0.0)
    for epoch in range(5):
        total_loss = 0.0

        for x_raw, y_raw in dataset:
            # convert dataset values into instances of Variable class
            x_vars = [Variable(v, label = 'x_i') for v, i in enumerate(x_raw)]
            y_vars = [Variable(v, label = 'y_i') for v, i in enumerate(y_raw)]

            # foward pass
            y_pred = net(x_vars)
            loss = loss_function(y_pred, y_vars)
            last_loss = loss
            
            # reset gradients for weights and biases
            for layer in net.layers:
                for neuron in layer.neurons:
                    for W in neuron.W:
                        W.grad = 0
                    neuron.b.grad = 0
            
            # backprop step
            loss.backprop()

            # update weights and biases
            eta = 0.05
            for layer in net.layers:
                for neuron in layer.neurons:
                    for W in neuron.W:
                        W.value = W.value - (eta * W.grad)
                    neuron.b.value = neuron.b.value - (eta * neuron.b.grad)

            # compute loss
            total_loss += loss.value
        
        # print training progress to terminal
        print(f"Epoch: {epoch}, Loss = {total_loss:.4f}")
    
    # for every weight, print the derivative chain for partial of loss with respect to weight
    chain_strings = []
    for layer in net.layers:
        for neuron in layer.neurons:
            # Derivative chains for weights
            for W in neuron.W:
                print(f"dLoss/d{W.label} =")
                chains = last_loss.parent_chain(W)
                
                for chain in chains:
                    seen = set()
                    labels = []
                    for p in chain:
                        if hasattr(p, "label") and p.label:
                            labels.append(p.label)
                            seen.add(p.label)
                    string = "Loss -> " + " -> ".join(labels)
                    print(string)
                    chain_strings.append(string)

    
if __name__ == "__main__":
    main()
import math     # import for exp in sigmoid
import random   # import to initilize network weights to random values

# class for variables in neural network
class Variable:
    def __init__(self, value, prev = (), op = ''):
        self.value = value    # value of variable
        self.grad = 0.0       # gradient value
        self.prev = set(prev) # parent values
        self.op = op          # operation that produced value from parents
        self.backward = lambda: None
    
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

# sigmoid function
def sigmoid(x):
    sig_value = 1 / (1 + math.exp(-x.value))
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
    def __init__(self, n_inputs):
        self.W = [Variable(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Variable(0.0) 

    # compute activation for neuron
    def __call__(self, x):
        weighted_sum = self.b
        for wi, xi in zip(self.W, x):
            weighted_sum = weighted_sum + wi * xi
        return sigmoid(weighted_sum)

# class for MLP layer  
class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = []
        for _ in range(n_outputs):
            neuron = Neuron(n_inputs)
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
            layer = Layer(sizes[i], sizes[i + 1])
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

def main():
    # create MLP
    net = MLP(n_inputs = 5, hidden_sizes = [6, 6], n_outputs = 3)

    dataset = xor_dataset()
    dataset = two_bit_adder_dataset()
  
    # train MLP
    for epoch in range(15000):
        total_loss = 0.0

        for x_raw, y_raw in dataset:
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
            eta = 0.1
            for layer in net.layers:
                for neuron in layer.neurons:
                    for W in neuron.W:
                        W.value = W.value - (eta * W.grad)
                    neuron.b.value = neuron.b.value - (eta * neuron.b.grad)

            # compute loss
            total_loss += loss.value
        
        # print training progress to terminal
        print(f"Epoch: {epoch}, Loss = {total_loss:.4f}")
    
    # make predictions
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

if __name__ == "__main__":
    main()



# x = Variable(-2.0)
# y = Variable(5.0)
# z = Variable(-4.0)

# q = x + y
# f = q * z

# f.backprop()

# print(f"f = {f.value}, df/dx = {x.grad}, df/dy = {y.grad}, df/dq = {q.grad}, df/df = {f.grad}")


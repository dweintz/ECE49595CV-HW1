import math

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












x = Variable(-2.0)
y = Variable(5.0)
z = Variable(-4.0)

q = x + y
f = q * z

f.backprop()

print(f"f = {f.value}, df/dx = {x.grad}, df/dy = {y.grad}, df/dq = {q.grad}, df/df = {f.grad}")
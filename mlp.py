import numpy as np
import time

'''
code should be able to control the number of input layers n, the input and output dimensions, the number of hidden units in each layer, and the learning rate. I can choose how to initialize gradient descent and number of iterations.

create two data sets:

xor: 2 inputs and 1 output
two-bit binary adderL five inputs and three outputs

create train and test splits

evauluate loss
'''


# class to represent dual numbers for automatic differentiation
class dual_number:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    # overloading
    def __pos__(self): return self # identity
    def __add__(self, y): return plus(self, y) # addition 
    def __radd__(self, x): return plus(x, self) # addition
    def __mul__(self, y): return times(self, y) # multiplication
    def __rmul__(self, x): return times(x, self) # multiplication
    def __repr__(self): return dual_number_to_str(self) # string representation

# return the primal of a dual number
def primal(x):
    if isinstance(x, dual_number): return x.a
    else: return x

# return the tangent of a dual number
def tangent(x):
    if isinstance(x, dual_number): return x.b
    else: return 0

# convert a number to dual form
def lift(x):
    if isinstance(x, dual_number): return x
    else: return dual_number(x, 0)

# add dual numbers
def plus(x, y):
    if isinstance(x, dual_number):
        if isinstance(y, dual_number):
            a = primal(x)
            b = tangent(x)
            c = primal(y)
            d = tangent(y)
            return dual_number(a + c, b + d)
        else: return x + lift(y)
    else:
        if isinstance(y, dual_number): return lift(x) + y
        else: return x + y

# multiply dual numbers
def times(x, y):
    if isinstance(x, dual_number):
        if isinstance(y, dual_number):
            a = primal(x)
            b = tangent(x)
            c = primal(y)
            d = tangent(y)
            return dual_number(a * c, a * d + b * c)
        else: return x * lift(y)
    else:
        if isinstance(y, dual_number): return lift(x) * y
        else: return x * y

# dual numbers to string
def dual_number_to_str(x):
    a = primal(x)
    b = tangent(x)
    if b >= 0:
        return "%s+%s*e"%(str(a), str(b))
    else:
        return "%s-%s*e"%(str(a), str(-b))

# define epsilon    
e = dual_number(0, 1)

# derivative
def derivative(f):
    return lambda x: tangent(f(x + 1 * e))

# replace the i-th element of x with xi
def replace_ith(x, i, xi):
    return [xi if j == i else x[j] for j in range(len(x))]

# partial derivative (derivative with respect to i-th variable)
def partial_derivative(f, i):
    return lambda x: derivative(lambda xi: f(replace_ith(x, i, xi)))(x[i])

# gradient (vector of all partial derivatives)
def gradient(f):
    return lambda x: [partial_derivative(f, i)(x) for i in range(len(x))]

# gradient descent
def naive_gradient_descent(f, x0, n, eta):
    x = x0
    for _ in range(n):
        x = [xi - eta * dfdxi for xi, dfdxi in zip(x, gradient(f)(x))]
    return x



# def f(x):
#     return 2 * x * x * x + 4 * x

# print(f(2))
# print(derivative(f)(2))
# print(derivative(derivative(f))(2))

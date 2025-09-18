import math
import random

class cobundle:
    def __init__(self, prim, tape):
        self.prim = prim
        self.tape = tape
    def __pos__(self): return self
    def __neg__(self): return 0-self
    def __add__(self, y): return plus(self, y)
    def __radd__(self, x): return plus(x, self)
    def __sub__(self, y): return minus(self, y)
    def __rsub__(self, x): return minus(x, self)
    def __mul__(self, y): return times(self, y)
    def __rmul__(self, x): return times(x, self)
    def __truediv__(self, y): return divide(self, y)
    def __rtruediv__(self, x): return divide(x, self)
    def __lt__(self, x): return lt(self, x)
    def __le__(self, x): return le(self, x)
    def __gt__(self, x): return gt(self, x)
    def __ge__(self, x): return ge(self, x)
    def __eq__(self, x): return eq(self, x)
    def __ne__(self, x): return ne(self, x)

class tape:
    def __init__(self, factors, tapes, fanout, cotg):
        self.factors = factors
        self.tapes = tapes
        self.fanout = fanout
        self.cotg = cotg

def cobun(x, factors, tapes): return cobundle(x, tape(factors, tapes, 0, 0))
def variable(x): return cobun(x, [], [])

def determine_fanout(t):
    t.fanout += 1
    if t.fanout == 1:
        for p in t.tapes: determine_fanout(p)

def initialize_cotg(t):
    t.cotg = 0
    t.fanout -= 1
    if t.fanout == 0:
        for p in t.tapes: initialize_cotg(p)

def reverse_sweep(cotg_val, t):
    t.cotg += cotg_val
    t.fanout -= 1
    if t.fanout == 0:
        cotg_val = t.cotg
        for factor, parent in zip(t.factors, t.tapes):
            reverse_sweep(cotg_val * factor, parent)

def cotg(y, x):
    if isinstance(y, cobundle):
        determine_fanout(y.tape)
        initialize_cotg(y.tape)
        determine_fanout(y.tape)
        reverse_sweep(1, y.tape)
        return cotg(y.prim, x)
    else:
        if isinstance(x, list): return [xi.tape.cotg for xi in x]
        else: return x.tape.cotg

def lift_real_to_real(f, dfdx):
    def me(x):
        if isinstance(x, cobundle):
            return cobun(me(x.prim), [dfdx(x.prim)], [x.tape])
        else: return f(x)
    return me

def lift_real_cross_real_to_real(f, dfdx1, dfdx2):
    def me(x1, x2):
        if isinstance(x1, cobundle):
            if isinstance(x2, cobundle):
                return cobun(me(x1.prim, x2.prim),
                             [dfdx1(x1.prim, x2.prim), dfdx2(x1.prim, x2.prim)],
                             [x1.tape, x2.tape])
            else:
                return cobun(me(x1.prim, x2), [dfdx1(x1.prim, x2)], [x1.tape])
        else:
            if isinstance(x2, cobundle):
                return cobun(me(x1, x2.prim), [dfdx2(x1, x2.prim)], [x2.tape])
            else: return f(x1, x2)
    return me

def lift_real_cross_real_to_boolean(f):
    def me(x1, x2):
        if isinstance(x1, cobundle): return me(x1.prim, x2)
        if isinstance(x2, cobundle): return me(x1, x2.prim)
        return f(x1, x2)
    return me

plus = lift_real_cross_real_to_real(lambda x1,x2: x1+x2, lambda x1,x2:1, lambda x1,x2:1)
minus = lift_real_cross_real_to_real(lambda x1,x2: x1-x2, lambda x1,x2:1, lambda x1,x2:-1)
times = lift_real_cross_real_to_real(lambda x1,x2: x1*x2, lambda x1,x2:x2, lambda x1,x2:x1)
divide = lift_real_cross_real_to_real(lambda x1,x2: x1/x2, lambda x1,x2:1/x2, lambda x1,x2:-x1/(x2*x2))
lt = lift_real_cross_real_to_boolean(lambda x1,x2:x1<x2)
le = lift_real_cross_real_to_boolean(lambda x1,x2:x1<=x2)
gt = lift_real_cross_real_to_boolean(lambda x1,x2:x1>x2)
ge = lift_real_cross_real_to_boolean(lambda x1,x2:x1>=x2)
eq = lift_real_cross_real_to_boolean(lambda x1,x2:x1==x2)
ne = lift_real_cross_real_to_boolean(lambda x1,x2:x1!=x2)
exp = lift_real_to_real(math.exp, lambda x: exp(x))

def derivative(f):
    def me(x):
        if isinstance(x, list):
            x_rev = [variable(xi) for xi in x]
        else: x_rev = variable(x)
        return cotg(f(x_rev), x_rev)
    return me

gradient = derivative

def sigmoid(x):
    def f(u): return 1 / (1 + math.exp(-u))
    def dfdx(u): s=f(u); return s*(1-s)
    return lift_real_to_real(f, dfdx)(x)

def flatten_params(params):
    flat = []
    for p in params:
        if isinstance(p, list): flat.extend(flatten_params(p))
        else: flat.append(p)
    return flat

class MLP:
    def __init__(self, layer_sizes, learning_rate):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.W = []
        self.b = []
        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            self.W.append([[variable(random.uniform(-0.1,0.1)) for _ in range(in_dim)] for _ in range(out_dim)])
            self.b.append([variable(0.0) for _ in range(out_dim)])

    def forward(self, x):
        a = [variable(xi) for xi in x]
        for W, b in zip(self.W, self.b):
            z = []
            for row, bias in zip(W, b):
                z_neuron = sum(w*a_i for w,a_i in zip(row,a)) + bias
                z.append(sigmoid(z_neuron))
            a = z
        return a

    def parameters(self):
        return self.W + self.b

def mse(pred, target):
    return sum((p-t)*(p-t) for p,t in zip(pred,target)) / len(pred)

def train(mlp, dataset, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in dataset:
            x = [x1, x2]
            target = [y]

            # Forward pass
            output = mlp.forward(x)

            # Loss as cobundle (do NOT use .prim)
            # loss = sum((o - t)*(o - t) for o, t in zip(output, target)) / len(target)
            loss = sum(((o - t)*(o - t) for o, t in zip(output, target)), start=variable(0.0)) / len(target)

    
            # Compute gradient
            for p in flatten_params(mlp.parameters()):
                g = cotg(loss, p)
                if isinstance(p.prim, list):
                    if isinstance(p.prim[0], list):
                        for i in range(len(p.prim)):
                            for j in range(len(p.prim[i])):
                                p.prim[i][j] -= mlp.learning_rate * g
                    else:
                        for i in range(len(p.prim)):
                            p.prim[i] -= mlp.learning_rate * g
                else:
                    p.prim -= mlp.learning_rate * g

            total_loss += float(loss.prim)  # convert loss to float for reporting

        print(f"Epoch {epoch+1}, Loss={total_loss/len(dataset):.4f}")

def xor(n_samples):
    dataset = []
    for _ in range(n_samples):
        x1 = round(random.random(), 2)
        x2 = round(random.random(), 2)
        y = int((x1 > 0.5) ^ (x2 > 0.5))
        dataset.append((x1, x2, y))
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    dataset = xor(200)
    mlp = MLP([2,3,1], learning_rate=0.5)
    train(mlp, dataset, epochs=200)

    num_correct = 0
    total = 0
    print("\nPredictions:")
    for x1,x2,y in dataset:
        output = mlp.forward([x1,x2])
        print(f"Input: {x1},{x2} | Expected: {y} | Pred: {round([o.prim for o in output][0], 0)} | {[o.prim for o in output]}")
        
        pred = int(round([o.prim for o in output][0], 0))
        expected = y

        if(pred == expected):
            num_correct += 1
        total += 1
        
    print(f"\nAccuracy = {round(num_correct / total * 100, 2)}%")
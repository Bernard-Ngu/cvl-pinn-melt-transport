"""
Minimal reverse-mode automatic differentiation on top of numpy.

PyTorch/JAX are not installable in this environment, so the PINN uses this
tape-based engine.  It supports exactly the operations the residuals need.

Design note
-----------
The PINN is written in a *mixed* (first-order) form: velocity, pressure,
compaction pressure, porosity, deviatoric stress and Darcy flux are all
network outputs, so every PDE residual contains only FIRST derivatives of
the network with respect to its inputs.  Those input-derivatives are
propagated forward analytically through the MLP (see model.mlp_with_jac),
and the resulting expressions are ordinary functions of the weights, which
this engine differentiates in reverse mode.
"""
import numpy as np


def _unbroadcast(g, shape):
    """Sum a gradient down to `shape` after numpy broadcasting."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class Tensor:
    __slots__ = ("v", "grad", "_parents", "_back", "requires_grad")
    __array_priority__ = 1000.0     # keep numpy from hijacking ndarray + Tensor

    def __init__(self, v, parents=(), back=None, requires_grad=False):
        self.v = np.asarray(v, dtype=np.float64)
        self.grad = None
        self._parents = parents
        self._back = back
        self.requires_grad = requires_grad or any(
            isinstance(p, Tensor) and p.requires_grad for p in parents)

    # ------------------------------------------------------------- helpers
    @property
    def shape(self):
        return self.v.shape

    def _wrap(self, v, parents, back):
        return Tensor(v, parents, back)

    # ---------------------------------------------------------- arithmetic
    def __add__(self, o):
        ov = o.v if isinstance(o, Tensor) else np.asarray(o, dtype=np.float64)
        out = Tensor(self.v + ov, (self, o),
                     lambda g, s=self, o=o: (
                         _unbroadcast(g, s.shape),
                         _unbroadcast(g, o.shape) if isinstance(o, Tensor) else None))
        return out

    __radd__ = __add__

    def __neg__(self):
        return Tensor(-self.v, (self,), lambda g: (-g,))

    def __sub__(self, o):
        return self + (-o if isinstance(o, Tensor) else -np.asarray(o, dtype=np.float64))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        ov = o.v if isinstance(o, Tensor) else np.asarray(o, dtype=np.float64)
        return Tensor(self.v * ov, (self, o),
                      lambda g, s=self, o=o, ov=ov: (
                          _unbroadcast(g * ov, s.shape),
                          _unbroadcast(g * s.v, o.shape) if isinstance(o, Tensor) else None))

    __rmul__ = __mul__

    def __truediv__(self, o):
        ov = o.v if isinstance(o, Tensor) else np.asarray(o, dtype=np.float64)
        return Tensor(self.v / ov, (self, o),
                      lambda g, s=self, o=o, ov=ov: (
                          _unbroadcast(g / ov, s.shape),
                          _unbroadcast(-g * s.v / (ov * ov), o.shape)
                          if isinstance(o, Tensor) else None))

    def __rtruediv__(self, o):
        ov = np.asarray(o, dtype=np.float64)
        return Tensor(ov / self.v, (self,),
                      lambda g, s=self, ov=ov: (-g * ov / (s.v * s.v),))

    def __pow__(self, c):
        c = float(c)
        return Tensor(self.v ** c, (self,),
                      lambda g, s=self, c=c: (g * c * s.v ** (c - 1.0),))

    def __matmul__(self, o):
        return matmul(self, o)

    def __rmatmul__(self, o):
        return matmul(o, self)

    def __getitem__(self, idx):
        def back(g, s=self, idx=idx):
            gr = np.zeros_like(s.v)
            np.add.at(gr, idx, g)
            return (gr,)
        return Tensor(self.v[idx], (self,), back)

    # -------------------------------------------------------------- reduce
    def sum(self, axis=None, keepdims=False):
        shp = self.v.shape
        return Tensor(self.v.sum(axis=axis, keepdims=keepdims), (self,),
                      lambda g, shp=shp, axis=axis, keepdims=keepdims: (
                          np.broadcast_to(
                              g if (keepdims or axis is None)
                              else np.expand_dims(g, axis), shp).copy(),))

    def mean(self, axis=None):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis) / n

    # ---------------------------------------------------------- backward
    def backward(self):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen or not isinstance(t, Tensor):
                return
            seen.add(id(t))
            for p in t._parents:
                if isinstance(p, Tensor) and p.requires_grad:
                    build(p)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.v)
        for t in reversed(topo):
            if t._back is None or t.grad is None:
                continue
            grads = t._back(t.grad)
            for p, g in zip(t._parents, grads):
                if isinstance(p, Tensor) and p.requires_grad and g is not None:
                    p.grad = g if p.grad is None else p.grad + g


# ------------------------------------------------------------------- ops
def matmul(a, b):
    av = a.v if isinstance(a, Tensor) else np.asarray(a, dtype=np.float64)
    bv = b.v if isinstance(b, Tensor) else np.asarray(b, dtype=np.float64)

    def back(g, a=a, b=b, av=av, bv=bv):
        ga = g @ bv.T if isinstance(a, Tensor) else None
        gb = av.T @ g if isinstance(b, Tensor) else None
        return (ga, gb)

    return Tensor(av @ bv, (a, b), back)


def tanh(t):
    y = np.tanh(t.v)
    return Tensor(y, (t,), lambda g, y=y: (g * (1.0 - y * y),))


def exp(t):
    y = np.exp(t.v)
    return Tensor(y, (t,), lambda g, y=y: (g * y,))


def log(t):
    return Tensor(np.log(t.v), (t,), lambda g, t=t: (g / t.v,))


def sqrt(t):
    y = np.sqrt(t.v)
    return Tensor(y, (t,), lambda g, y=y: (g * 0.5 / y,))


def softplus(t, beta=1.0):
    x = beta * t.v
    y = np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0)))) / beta
    return Tensor(y, (t,), lambda g, x=x: (g / (1.0 + np.exp(-np.minimum(x, 30.0))),))


def concat(ts, axis=1):
    vs = [t.v for t in ts]
    sizes = [v.shape[axis] for v in vs]
    offs = np.cumsum([0] + sizes)

    def back(g, offs=offs, axis=axis):
        return tuple(np.take(g, range(offs[i], offs[i + 1]), axis=axis)
                     for i in range(len(sizes)))

    return Tensor(np.concatenate(vs, axis=axis), tuple(ts), back)


def where(mask, t, other=0.0):
    """Select with a *fixed* (non-differentiable) boolean mask."""
    ov = other.v if isinstance(other, Tensor) else other
    return Tensor(np.where(mask, t.v, ov), (t, other),
                  lambda g, m=mask, other=other: (
                      g * m,
                      g * (~m) if isinstance(other, Tensor) else None))


def sigmoid(t):
    return (tanh(t * 0.5) + 1.0) * 0.5


def mse(t):
    return (t * t).mean()

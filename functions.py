import numpy as np


def linear(x, a, b):
    return a*x+b


def quadratic(x, a, b, c):
    return a*x**2+b*x+c


def cubic(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d


def exp(x, a, b, c):
    return a * np.exp(b * x) + c


def root(x, a, b, c):
    return a*(b*x)**0.5+c


def sin(x, a, b, c, d):
    return a*np.sin(b*x+c)+d


FUNCS = [linear, quadratic, cubic, exp, root, sin]

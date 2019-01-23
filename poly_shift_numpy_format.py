import random

import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank

from scipy.linalg import lu
from scipy.linalg import pascal

import matplotlib.pyplot as plt

import sympy
import sympy as sp
from sympy.abc import x

from IPython.display import display, Math, Latex

#
# Display Numpy Poly1d's
#

def plot_polys(polys, a=-10, b=10, steps=100):
    sp.init_printing()
    from sympy.abc import x
    X = np.linspace(a,b,steps)

    for p in polys:
        y = p(X)
        l = sp.Poly(p.c,x).as_expr()
        plt.plot(X,y, label=l)
    
    plt.grid()
    plt.legend()
    plt.show()
    

#
# create Horizontal Shift Matrix
#

def poly_shift_matrix(order, h):
    # h -- horizontal shift amount
    
    # n+1 coeficients to solve for -- n+1 x n+1 shift matrix
    n = order + 1
    
    # pascal matrix
    P = pascal(n, kind='upper')
    
    # create H
    H = np.eye(n)
    _h = 1
    for k in range(n):
        for i in range(n-k):            
            H[i,i+k] = _h
        _h *= h

    # create shift matrix - flip for numpy convention
    S = np.flip(H * P)
    
    # return
    return S

#
# Horizontal Shift
#

def poly_horizontal_translation(p, delta_x):
    
    # r(x) = p(x + delta_x)
    S = poly_shift_matrix(p.order,delta_x)
    r = np.poly1d( S @ p )
    
    # return r(x) -- numpy polynomial
    return r

#
# Vertical Shift
#

def poly_vertical_translation(poly, delta_y):
    # r(x) = p(x) + delta_y
    r = poly + delta_y
    return r 


#
# Translate Numpy Polynomial
#

def translate_polynomial(p, delta_x=0, delta_y=0):
    r = p
    
    if delta_x != 0:
        # r(x) = p(x + delta_x)
        r = poly_horizontal_translation(r, delta_x)
        
    if delta_y != 0:
        # r(x) = p(x) + delta_y
        r = poly_vertical_translation(r, delta_y)
        
    return r

#
# Random Testing
#

# SETTINGS
MIN_DEGREE = 2
MAX_DEGREE = 5
SHIFT_RANGE = 3 # MAX

# generate random base polynomial
n = random.randint(MIN_DEGREE,MAX_DEGREE)
p = np.poly1d( np.random.randn(n))

# translate base polynomial
x_shift, y_shift = random.randint(-SHIFT_RANGE,SHIFT_RANGE), random.randint(-SHIFT_RANGE,SHIFT_RANGE)
q = translate_polynomial(p, delta_x=x_shift)
r = translate_polynomial(p, delta_y=y_shift)

# plot results
plot_polys([p, q, r], a=-3, b=3)


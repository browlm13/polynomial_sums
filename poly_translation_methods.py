import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank
from scipy.linalg import lu
from scipy.linalg import pascal
import sympy
import matplotlib.pyplot as plt

#
# Display Numpy Poly1d's
#

def plot_polys(polys,names=None, a=-10, b=10, steps=100):

    x = np.linspace(a,b,steps)

    if names == None:
        for p in polys:
            y = p(x)
            plt.plot(x,y)
    
    else:
        for p,n in zip(polys,names):
            y = p(x)
            plt.plot(x,y,label=n)
    plt.grid()
    plt.legend()
    plt.show()
        
#
# define a translation matrix Lh of size nxn - horizontal translation
#

def poly_translation_matrix(n, delta_x, kind='lower'):

    # pascal matrix
    P = pascal(n, kind='lower')

    # exponent matrix
    L = np.tril(np.ones(shape=(n,n)))
    R = np.tril(matrix_power(L, 2)-1)

    # D matrix, delta_x's raised to corresponding powers
    D = np.tril(np.power(-delta_x,R)) 

    # Create S poly shift matrix with offset delta_x
    Ld = (D * P) 

    if kind == 'lower':
        return Ld

    # return upper triangular
    Ud = Ld.T
    return Ud

#
# define a translation matrix of size nxn - vertical translation
#

def poly_vertical_translation(poly, delta_y):
    poly += delta_y
    return poly 

def poly_horizontal_translation(p, delta_x):
    # r(x) = p(x + delta_x)
    Uh = poly_translation_matrix(len(p)+1, delta_x, kind='upper')
    coefs = Uh @ p.c[::-1]
    q = np.poly1d( coefs[::-1] )
    return q

def translate_polynomial(p, delta_x=0, delta_y=0):
    r = p
    
    if delta_x != 0:
        # r(x) = p(x + delta_x)
        r = poly_horizontal_translation(r, delta_x)
        
    if delta_y != 0:
        # r(x) = p(x) + delta_y
        r = poly_vertical_translation(r, delta_y)
        

    return r


p1 = np.poly1d( np.array([1,0,0]))
p2 = translate_polynomial(p1, delta_x=2)
p3 = translate_polynomial(p1, delta_y=5)
plot_polys([p1, p2,p3], names=['p1', 'p2', 'p3'], a=-5, b=5)


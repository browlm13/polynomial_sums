

import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank
from scipy.linalg import lu
from scipy.linalg import pascal
import sympy

# define a translation matrix Lh of size nxn
def poly_translation_matrix(n, h, kind='lower'):

	# pascal matrix
	P = pascal(n, kind='lower')

	# exponent matrix
	L = np.tril(np.ones(shape=(n,n)))
	R = np.tril(matrix_power(L, 2)-1)

	# H matrix, H's raised to corresponding powers
	H = np.tril(np.power(-h,R)) 

	# Create S poly shift matrix with offset h
	Lh = (H * P) 

	if kind == 'lower':
		return Lh

	# return upper triangular
	Uh = Lh.T
	return Uh

def translate_polynomial(p, h):

	# q(x) = p(x + h)
	Uh = poly_translation_matrix(len(p)+1,h, kind='upper')
	print(Uh)
	coefs = Uh @ p.c[::-1]
	q = np.poly1d( coefs[::-1] )

	return q

def poly_translation_poly(p, h):


	n = len(p)+1

	Uh = poly_translation_matrix(n,h, kind='upper')
	I = np.eye(n)
	#A = Uh - I # for unshifted

	A = np.linalg.inv(Uh) @ (Uh - I) # for pn-1(x)

	coefs = A @ p.c[::-1]
	q = np.poly1d( coefs[::-1] )


	return q

def padded_trans_mat(n, h, depth):

	M = np.zeros(shape=(n,n))
	Uh = poly_translation_matrix(n-depth,h, kind='upper')

	M[0:-depth, 0:-depth] = Uh

	return M



#
# Testing
#

shift_amount = 1

p = np.poly1d( np.array([1/3,1/2,1/6,0]))
p_shift = translate_polynomial(p, shift_amount)

q = poly_translation_poly(p, shift_amount)
q_shift = translate_polynomial(q, shift_amount)


r = poly_translation_poly(q, shift_amount)
r_shift = translate_polynomial(r, shift_amount)

s = poly_translation_poly(r, shift_amount)
s_shift = translate_polynomial(s, shift_amount)

print(p)
print(q)
print(r)
print(s)

print(p_shift - p)

n= 5
h = 1
Uh = poly_translation_matrix(n,h, kind='upper')


print(np.linalg.inv(Uh))

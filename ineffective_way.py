

"""

	Finding the formula for,
		sum of first n natural numbers...

	     (*) q(n, d=1) = n^2/2 + n/2


	  	We will use...

			sum of first n squares (unkown, recursive definition)...

				  		    q(n+1, d=2) = q(n, d=2) + (n+1)^2

				  then,

				  		(1)  q(n+1, 2) - q(n, 2)  =  (n+1)^2

			(1)  q(n+1, 2) - q(n, 2)  =  (n+1)^2


			expansion of squares...

				(x + 1)^2 =  x^2  +  2x    + 1

				(2) (n + 1)^2 = n^2  + 2n +1

			(2) (n + 1)^2 = n^2  + 2n +1



		Derivation...



	          	c0               c1            c2               c3
	         ----------------------------------------------------
	 1^2  =   (0 + 1)^2     =    0^2         +  2*0              + 1
	 2^2  =   (1 + 1)^2     =    1^2         +  2*1              + 1
	 3^2  =   (2 + 1)^2     =    2^2         +  2*2              + 1
	 ...
+(n+1)^2  =   (n + 1)^2     =    n^2         +  2*n              + 1 
----------------------------------------------------------------------
			   q(n+1, 2)    =   q(n, 2)      +  2*q(n, 1)  	     + n+1
			   									2*(TARGET)

			Subtract c1 from both sides,

				q(n+1, 2) - q(n, 2) =     2*q(n, 1)  + n+1

			Substitute equation (1) for the left hand side,

				(n+1)^2 =  2*q(n, 1)  + n+1

			And (2) for the left hand side again so it looks clean,
				n^2  + 2n +1 = 2*q(n, 1)  + n+1

			We can now solve for q(n, 1),

				q(n, 1) = 1/2[n^2  + n]
				q(n, 1) = n^2/2 + n/2 


	This process can be used to derive a formula for q(n, d) for d = 2,3,4,5 ...


"""

import numpy as np
from scipy.special import comb


def q( n, d=1 ):
	""" q( n ) - q( n-1 ) = n**d """

	if np.isscalar(n):
		# must be an integer
		n = int(n)
		s = 0
		for i in range(1,n+1): 
			s += i**d
		return s

	return np.array([ q( ni, d=d ) for ni in n ])

def compute_C(max_d):

	# solve to max_d
	C = np.zeros(shape=(max_d+1, max_d+2))

	# initialize 0th row of C as formula for q_0(n) which is (1)*n^(1) as a polynomial
	C[0,1] = 1.0 # first column represents x^1 or n^1


	# include max_d, but not max_d+1
	for a in range(1,max_d+1):

		# (-1)*n**0
		C[a,0] += -1.0

		# include a-1, but not a
		for k in range(a):
			c = comb(a+1, k, exact=True)
			C[a,k] += c
			C[a,:] += -c*C[k,:]

		# (a+1 choose a)*n**a
		C[a,a] += comb(a+1, a, exact=True)

		# (a+1 choose a+1)*n**(a+1)
		C[a,a+1] += 1.0

		# (a+1 choose a)**(-1) 
		C[a,:] *= 1.0/comb(a+1, a, exact=True)


	# return coefficient matrix
	return C


def get_pd(d, C):

	p = np.poly1d(C[d,:][::-1])
	return p


# method 2 - induction coef solver -- direct
def get_coefs(d):

	A = np.zeros(shape=(d+2, d+2))
	b = np.zeros(shape=(d+2,))

	# fill b
	for i in range(len(b)):
		b[i] = comb(d, i, exact=True)

	print(b)

	# fill a
	for r in range(d+2):
		# c choose r
		for c in range(r+1, d+2):
			A[r,c] = comb(c, r, exact=True)

	A = A[:-1,1:]
	b = b[:-1]

	# solve system
	x = np.linalg.solve(A,b)
	x1 = np.zeros(shape=(len(x)+1,))
	x1[1:,] = x
	x = x1

	return np.poly1d(x[::-1])


def get_A(d):

	A = np.zeros(shape=(d+2, d+2))

	# first row comes from constraint (1) 
	A[0,:] = np.ones(shape=(d+2))

	# remaining rows come from constraint (2) equations
	# however a more efficient method can be used to fill the matrix

	# row 1 has a leading 0 then trailing 1's
	A[1,1:] = np.ones(shape=(d+1))

	# the diagnol elements are sums of the previous colums
	# 2 elements, one on the same row and one on the previous row

	# the main diagnol is filled with 1, 2, ..., d+2
	for i in range(2, d+2):
		A[i,i] = i

	# fill the remaining elements row by row
	for r in range(2, d+2):
		for c in range(r+1, d+2):
			A[r,c] = A[r-1,c-1] + A[r,c-1]


	return A


import math
def get_b(d):

	b = np.zeros(shape=(d+1,))

	d_factorial = math.factorial(d)
	i_factorial = 1

	b[0] = 1
	for i in range(1,d+1):

		# d choose i
		i_factorial *= i

		b[i] = d_factorial/i_factorial/math.factorial(d-i)

	v = np.ones(shape=(d+2,))
	v[1:] = b
	
	return v


def get_p(d):

	A = get_A(d)
	b = get_b(d)

	print(A)
	print(b)

	# solve system
	x = np.linalg.solve(A,b)

	#print(x)


	return np.poly1d(x[::-1])


#
# Testing
#


if __name__ == "__main__":

	#get_A(6)
	#get_b(6)
	#get_coefs(6)



	

	# Settings -- Check until q_d(n), for n=1,2,...,MAX_N
	MAX_D = 5
	MAX_N = 10

	# generate test values
	ns = np.linspace(1,MAX_N,MAX_N)

	# compute polynomials coeffient matrix
	C = compute_C(MAX_D)

	# check against known formula 
	for d in range( MAX_D+1):

		# compute actual values using known formula
		actual = q(ns, d=d)

		# compute values using derived polynomials
		#p = get_pd(d,C)
		#p =  get_coefs(d)
		p = get_p(d)
		derived = p(ns)

		print(p)

		# compute the total error
		error = abs(np.sum(np.array(actual-derived)))

		# display results
		#print("Known forumla results: ", actual)
		#print("Derived formula results: ", derived)
		print("Total Error: ", error)






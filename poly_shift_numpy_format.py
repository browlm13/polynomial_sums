#
# create Horizontal Shift Matrix
#

def poly_shift_matrix(n, h):
    
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
    S = H * P
    S = np.flip(S)
    
    return S

def poly_horizontal_translation(p, delta_x):
    
    # r(x) = p(x + delta_x)
    n = p.order +1
    S = poly_shift_matrix(n,delta_x)
    r = np.poly1d( S @ p )
    
    return r

p = np.poly1d( np.array([1,0,0]))
delta_x = 3
r = poly_horizontal_translation(p, delta_x)

plot_polys([p, r], a=-2, b=2)

print(r(-delta_x) - p(0))

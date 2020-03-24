import numpy as np


####################################################################################################
# Exercise 1: Interpolation
# https://www.lernhelfer.de/schuelerlexikon/mathematik-abitur/artikel/newtonsches-und-lagrangesches-interpolationsverfahren
def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0) 
    base_functions = []
    
    # TODO: Generate Lagrange base polynomials and interpolation polynomial
    n = x.shape[0]
    for j in range(n):  # For each L: L_0, L_1, L_2, ... 
        l = 1  # For each L_j is l=1 at the beginning.
        for s in range(n):  # Computes each factor of the product L_j. There are n factors. 
            if s != j:
                l *= np.poly1d([1, -x[s]]) / (x[j] - x[s])  # l is L_j at the end. 
        base_functions.append(l)  # Adds each L_j to base_functions. 
    
    """
    Example: degree = 3
    L_0 = ((x - x_1)(x - x_2)(x - x_3)) / ((x_0 - x_1)(x_0 - x_2)(x_0 - x_3))
    L_1 = ((x - x_0)(x - x_2)(x - x_3)) / ((x_1 - x_0)(x_1 - x_2)(x_1 - x_3))
    ...
    """
    
    p = 0
    for j in range(n):  # Computes the lagrange interpolation.
        p += y[j] * base_functions[j]  # L(x) = y_0*L_0 + y_1*L_1 + y_2*L_2 + ... + y_n*L_n
    
    polynomial = np.poly1d(p)                    
    
    return polynomial, base_functions
    
# https://homepage.divms.uiowa.edu/~atkinson/ftp/ENA_Materials/Overheads/sec_4-3.pdf    
def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points / Ableitungspunkte

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    
    # TODO compute piecewise interpolating cubic polynomials

    n = len(x) - 1  # We have len(x) points and need (#points-1) functions.
    
    for k in range(n):  # On each of the intervals [x_0,x_1], [x_1,x_2], [x_2,x_3],...
        
        matrix = np.zeros((4,4))  # Set up the matrix in the following.
        
        """
        The matrix should look like this:
        1 x_0 x_0^2 x_0^3
        1 x_1 x_1^2 x_1^3
        0 1   2*x_0 3*x_0^2
        0 1   2*x_1 3*x_1^2
        """
        
        # First the cubic polynomials:
        for i in range(2):
            for j in range(4):
                if (j == 0 and i == 0) or (i == 1 and j == 0):
                    matrix[i,j] = 1
                if i == 0:
                    matrix[i,j] = x[k]**j
                if i == 1:
                    matrix[i,j] = x[k+1]**j
        
        # Then the linear polynomials/derivations:            
        for l in range(2):
            for m in range(4):
                if m == 0:
                    matrix[l+2,m] = 0
                if m == 1:
                    matrix[l+2,m] = 1
                if l == 0 and m > 1:
                    matrix[l+2,m] = m*x[k]**(m-1)
                if l == 1 and m > 1:
                    matrix[l+2,m] = m*x[k+1]**(m-1)
        
        f = np.array([y[k], y[k + 1], yp[k], yp[k + 1]])  # We determine the constants by using f(x_k) = f_k = y_k and f'(x_k) = f'_k = yp_k.
        coeff = np.linalg.solve(matrix, f)  # Computes the c-coefficients.
        coeff = np.flipud(coeff)
        polynomial = np.poly1d(coeff)  # The first polynomial schould look like this: 1*c_00 + x[0]*c_01 + (x[0]^2)*c_02 + (x[0]^3)*c_03 = f_0
        
        spline.append(polynomial)
                
    return spline


####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    
    # TODO construct linear system with natural boundary conditions
    x_points = x.size  # n points -> (n-1) cubic functions, each has 4 degrees of freedom -> 4(n-1) variables
    matrix = np.zeros((4 * (x_points - 1), 4 * (x_points - 1)))  # base case: linear equations + 1. and 2. derivation: 2*(n-1) + 2(n-2), 2 left over
    f = np.zeros((4 * (x_points - 1), ))
    polynomial = np.poly1d(1)
    
    for k in range(x_points - 2):  # Builds the (4n-8)x(4n-4) part of the matrix.
        auxiliary_function = np.zeros((4, 8))
        """
        First the cubic polynomials:
        1 x_0 x_0^2 x_0^3 0 0 0 ...
        1 x_1 x_1^2 x_1^3 0 0 0 
        """
        auxiliary_function[0] = [1, x[k], x[k]**2, x[k]**3, 0, 0, 0, 0]
        auxiliary_function[1] = [1, x[k + 1], x[k + 1]**2, x[k+ 1 ]**3, 0, 0, 0, 0]
        """
        Then the linear polynomials/derivations: 
        0 1 2*x_1 3*x_1 0 -1 -2*x_1 -3*x_1^2 0 0 0 ...
        0 0 2   6*x_1   0 0  -2     -6*x_1   0 0 0 ...        
        """
        auxiliary_function[2] = [0, 1, 2*x[k + 1], 3*x[k + 1]**2, 0, -1, -2*x[k + 1], -3*x[k + 1]**2]
        auxiliary_function[3] = [0, 0, 2, 6*x[k + 1], 0, 0, -2, -6 * x[k + 1]]
        
        matrix[k * 4: k * 4 + 4, k * 4: k * 4 + 8] = auxiliary_function  # The indices are: [0:4,0:8], [4:8,4:12], [8:12,8:16],... 
    
    # Natural boundary conditions:
    matrix[4 * (x_points - 1) - 1: , 4 * (x_points - 1) - 4: ] = [0, 0 , 2, 6 * x[x_points - 1]]
    matrix[4 * (x_points - 1) - 2: 4 * (x_points - 1) - 1, 0: 4 ] = [0, 0 , 2, 6 * x[0]]
    # The penultimate two lines:
    matrix[4 * (x_points - 1) - 3: 4 * (x_points - 1) - 2, 4 * (x_points - 1) - 4: ] = [1, x[x_points - 1], x[x_points - 1]**2, x[x_points - 1]**3]
    matrix[4 * (x_points - 1) - 4: 4 * (x_points - 1) - 3, 4 * (x_points - 1) - 4: ] = [1, x[x_points - 2], x[x_points - 2]**2, x[x_points - 2]**3]
    
    for k in range(x_points - 1):  # Builds vector with function values. It looks like this: [f_0, f_1, 0, 0, f_1, f_2, 0, 0, f_2, f_3,...]
        f[0 + k * 4] = y[k]
        f[1 + k * 4] = y[k + 1]
            
    # TODO solve linear system for the coefficients of the spline
    coeff = np.linalg.solve(matrix, f)
    
    spline = []

    # TODO extract local interpolation coefficients from solution
    for k in range(x_points - 1):
        polynomial = np.poly1d([coeff[4 * k + 3], coeff[4 * k + 2], coeff[4 * k + 1], coeff[4 * k]])
        spline.append(polynomial)
            
    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    
    # TODO: construct linear system with periodic boundary conditions
    x_points = len(x)  # n points -> (n-1) cubic functions, each has 4 degrees of freedom -> 4(n-1) variables
    matrix = np.zeros((4 * (x_points - 1), 4 * (x_points - 1)))  # base case: linear equations + 1. and 2. derivation: 2*(n-1) + 2(n-2), 2 left over
    f = np.zeros((4 * (x_points - 1), ))
    polynomial = np.poly1d(1)
    
    for k in range(x_points - 2):  # Builds the (4n-8)x(4n-4) part of the matrix.
        auxiliary_function = np.zeros((4, 8))
        """
        First the cubic polynomials:
        1 x_0 x_0^2 x_0^3 0 0 0 ...
        1 x_1 x_1^2 x_1^3 0 0 0 
        """
        auxiliary_function[0] = [1, x[k], x[k]**2, x[k]**3, 0, 0, 0, 0]
        auxiliary_function[1] = [1, x[k + 1], x[k + 1]**2, x[k+ 1 ]**3, 0, 0, 0, 0]
        """
        Then the linear polynomials/derivations: 
        0 1 2*x_1 3*x_1 0 -1 -2*x_1 -3*x_1^2 0 0 0 ...
        0 0 2   6*x_1   0 0  -2     -6*x_1   0 0 0 ...        
        """
        auxiliary_function[2] = [0, 1, 2*x[k + 1], 3*x[k + 1]**2, 0, -1, -2*x[k + 1], -3*x[k + 1]**2]
        auxiliary_function[3] = [0, 0, 2, 6*x[k + 1], 0, 0, -2, -6 * x[k + 1]]
        
        matrix[k * 4: k * 4 + 4, k * 4: k * 4 + 8] = auxiliary_function  # The indices are: [0:4,0:8], [4:8,4:12], [8:12,8:16],... 
    
    # The penultimate two lines:
    matrix[4 * (x_points - 1) - 4: 4 * (x_points - 1) - 3, 4 * (x_points - 1) - 4: ] = [1, x[x_points - 2], x[x_points - 2]**2, x[x_points - 2]**3]
    matrix[4 * (x_points - 1) - 3: 4 * (x_points - 1) - 2, 4 * (x_points - 1) - 4: ] = [1, x[x_points - 1], x[x_points - 1]**2, x[x_points - 1]**3]
    
    # The periodic boundary conditions:
    for k in range(1, 4):  # Builds the first line.
        matrix[4 * (x_points - 1) - 2, k] = k * (x[0]**(k - 1))  # [0, 1, 2x_0, 3x_0^2,...
        matrix[4 * (x_points - 1) - 2, k + 4 * (x_points - 1) - 4] = -(x[x_points - 1]**(k - 1)) * k  # ...0, -1, -2x_(n-1), -3x_(n-1)^2]
    
    # The last line:
    # [0, 0, 2, 6x_0, ...
    matrix[4 * (x_points - 1) - 1, 2] = 2
    matrix[4 * (x_points - 1) - 1, 3] = 6 * x[0]
    # ...0, 0, -2, -6x_(n-1)]
    matrix[4 * (x_points - 1) - 1, 4 * (x_points - 1) - 2] = -2
    matrix[4 * (x_points - 1) - 1, 4 * (x_points - 1) - 1] = -6 * x[x_points - 1]
    
    
    for k in range(x_points - 1):  # Builds vector with function values. It looks like this: [f0, f1, 0, 0, f1, f2, 0, 0, f2, f3,...]
        f[0 + k * 4] = y[k]
        f[1 + k * 4] = y[k + 1]
            
    # TODO solve linear system for the coefficients of the spline
    coeff = np.linalg.solve(matrix, f)
    
    spline = []

    # TODO extract local interpolation coefficients from solution
    for k in range(x_points - 1):
        polynomial = np.poly1d([coeff[4 * k + 3], coeff[4 * k + 2], coeff[4 * k + 1], coeff[4 * k]])
        spline.append(polynomial)

    return spline


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
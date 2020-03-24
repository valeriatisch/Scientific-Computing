import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

# https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    m, n = A.shape  # m=rows, n=columns of A
    v = b.shape[0]

    if n != m:
        raise ValueError("Matrix is not square.")

    if n != v:
        raise ValueError("Matrix and vector sizes are incompatible.")

    # TODO: Perform gaussian elimination

    if not use_pivoting:  # if pivoting is disabled:
        for i in range(n):
            if A[i, i] == 0:  # If one of the diagonal elements is 0, pivoting is necessary.
                raise ValueError("Pivoting is disabled but necessary.")
            for k in range(i + 1, n):
                m_ki = -A[k, i] / A[i, i]  # required factor for row i
                for j in range(i, n):
                    if i == j:
                        A[k, j] = 0
                    else:
                        A[k, j] += m_ki * A[i, j]
                b[k] += m_ki * b[i]

    if use_pivoting:
        for i in range(n):
            max_element = abs(A[i, i])
            max_row = i
            for j in range(i + 1, n): # Searchs for max element = pivot.
                if abs(A[j, i]) > max_element:
                    max_row = j
                    max_element = abs(A[j, i])
                else:
                    pass
            for k in range(i, n):  # Swaps rows in A.
                tmp_A = A[i, k]
                A[i, k] = A[max_row, k]
                A[max_row, k] = tmp_A
            tmp_b = b[i]  # Swaps rows in b.
            b[i] = b[max_row]
            b[max_row] = tmp_b
            for k in range(i + 1, n):  # Computes.
                m_ki = -A[k, i] / A[i, i]
                for p in range(i, n):
                    A[k, p] += m_ki * A[i, p]
                b[k] += m_ki * b[i]

    return A, b


# http://www.math.usm.edu/lambers/mat610/sum10/lecture4.pdf
def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    A, b = gaussian_elimination(A, b, True)

    m, n = A.shape  # m = rows, n = columns of A
    v = b.shape[0]

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    if n != v:
        raise ValueError("Matrix and vector sizes are incompatible.")

    # TODO: Initialize solution vector with proper size
    x = np.zeros(v)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist

    if A[n - 1, n - 1] == 0:
        raise ValueError

    x[n - 1] = b[n - 1] / A[n - 1, n - 1]

    for i in range(n - 1, -1, -1):
        if A[i, i] == 0 and b[i] != 0:
            raise ValueError("No solutions exist.")
        if A[i, i] == 0 and b[i] == 0:
            raise ValueError("Infinite solutions exist.")
        subsum = 0
        for j in range(i + 1, n):
            subsum += A[i, j] * x[j]
            x[i] = (b[i] - subsum) / A[i, i]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

# https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


# http://www.math.usm.edu/lambers/mat610/sum10/lecture4.pdf
def forward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()
    
    m, n = A.shape  # m = rows, n = columns of A
    v = b.shape[0]

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if n != m:
        raise ValueError("Matrix is not square.")
        
    if n != v:
        raise ValueError("Matrix and vector sizes are incompatible.")

    # TODO: Initialize solution vector with proper size
    x = np.zeros(v)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    # column oriented
    for i in range(m):
        if np.isclose(A[i, i], 0):
            raise ValueError("Infinite solutions exist.")
        if np.isclose(A[i, i], 0) and b[i] != 0:
            raise ValueError("No solutions exist.")
        x[i] = b[i]/A[i, i]
        A[i, i] = 1
        for j in range(i+1, m):
            b[j] -= A[j, i]*x[i]
            A[j, i] = 0
    
    return x

# https://www.uni-muenster.de/AMM/num/Vorlesungen/Numerik1_WS06/loesungen06/Prog_cholesky.pdf
def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape

    if m != n:
        raise ValueError("Matrix is not square.")

    if not check_symmetric(M, tol=1e-8):  # if M != M.T
        raise ValueError("Matrix is not symmetric.")



    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))
    
    for j in range(n):
        for i in range(n):
            if j == i:  # for diagonal elements
                subsum_d = 0
                for k in range(j):  # Builds the sum.
                    subsum_d += L[j, k]**2
                if (M[j, j] - subsum_d) < 0:  # Checks whether matrix is not pd.
                    raise ValueError("Matrix is not pd.")
                else:  # Computes all diagonal elements.
                    L[j, j] = (M[j ,j] - subsum_d)**(1/2)
            if j > i:  # for not diagonal elements
                subsum_nd = 0
                for k in range(i):
                    subsum_nd += L[j, k]*L[i, k]
                L[j, i] = (M[j, i] - subsum_nd)*(1/L[i, i])

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    v = b.shape[0]

    if n != m:
        raise ValueError("Matrix is not square.")

    if n != v:
        raise ValueError("Matrix and vector sizes are incompatible. / Sizes of L, b do not match.")

    if not np.allclose(L, np.tril(L)):
        raise ValueError("L is not lower triangular matrix.")


    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)

    x = back_substitution(L.T, forward_substitution(L, b))

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    n = n_shots*n_rays
    m = n_grid*n_grid
    L = np.zeros((n, m))
    # TODO: Initialize intensity vector
    v = n_rays*n_shots
    g = np.zeros(v)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

    for i in range(n_shots):  # Iterates over n shots.
        theta = (np.pi*i)/n_shots
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        # how many shots of one ray
        for j in range(n_rays):  # Sets up vector g.
            var_g = n_rays*i + j
            g[var_g] = intensities[j]  # Fills vector with intensities.

        for k in range(len(ray_indices)):  # Sets up system matrix L.
            var_L_0 = n_rays*i + ray_indices[k]
            var_L_1 = isect_indices[k]
            L[var_L_0, var_L_1] = lengths[k]

    return [L, g]


def compute_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)

    A = np.dot(L.T, L)
    b = np.dot(L.T, g)
    c = np.linalg.solve(A, b)

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    for k in range(n_grid):
        for j in range(n_grid):
            tim[k, j] = c[n_grid*k + j]

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
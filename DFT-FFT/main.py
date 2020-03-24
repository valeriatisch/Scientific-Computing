import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    
    # TODO: initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')
    
    # TODO: create principal term for DFT matrix
    auxiliary_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            auxiliary_matrix[i, j] = i * j

    # TODO: fill matrix with values
    omega = np.exp(-2 * np.pi * 1j / n)
    F = np.power(omega, auxiliary_matrix)
   
    # TODO: normalize dft matrix
    F /= np.sqrt(n)

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    unitary = True
    # TODO: check that F is unitary, if not return false
    n = matrix.shape[0]
    
    identity_matrix = np.identity(n)
    
    conjugate_transpose = matrix.conj().T
    
    mult = np.dot(matrix, conjugate_transpose)
    
    if np.allclose(mult, identity_matrix):
        unitary = True
    else:
        unitary = False

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals and extract harmonics out of DFT matrix
    DFT_matrix = dft_matrix(n)  # Constructs the DFT matrix.
    
    for i in range(n):  # Builds 128 discrete δ-impulses.
        e_i = np.zeros(n) # Each element is 0 except the i-th element.
        e_i[i] = 1
        sigs.append(e_i)
        
    for signal in sigs:
        transformed_DFT_matrix = np.dot(DFT_matrix, signal)  # Computes the Fourier transform.  
        fsigs.append(transformed_DFT_matrix)

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT
# https://stackoverflow.com/questions/699866/python-int-to-binary
# https://www.geeksforgeeks.org/reverse-string-python-5-different-ways/
def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # TODO: implement shuffling by reversing index bits
    
    n = data.shape[0]
    auxiliary_array = np.zeros(n, dtype = 'complex128')

    for i in range(n):  # For each index of data:
        binary_rep = "{0:b}".format(i)  # Converts int to binary.
        binary_rep = binary_rep.zfill(int(np.log2(n)))  # Fills the first (log2(n)-len(binary_rep)) spaces with zeros.
        # log2(2) = 1, log2(4) = 2, log2(8) = 3, log2(16) = 4,... is the width of binary numbers.  
        reversed_bin = binary_rep[::-1]  # Reverses the binary number.
        int_rep = int(reversed_bin, 2)  # Converts binary to int.
        element = data[i]
        auxiliary_array[int_rep] = element    
        
    """
    for i in range(n):
        data[i] = auxiliary_array[i]
    """
    
    data = auxiliary_array
    
    return data

# https://www.youtube.com/watch?v=htCj9exbGo0
def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # TODO: first step of FFT: shuffle data
    fdata = shuffle_bit_reversed_order(fdata)

    # TODO: second step, recursively merge transforms
    m = 0
    auxiliary_var = 1
    
    while(m < np.log2(n)):  # For all levels m of the tree:
        
        for k in range(2**m):  # For all values of k = [0,1,...,2^m[ on the current level:
            omega = np.exp((-1j * k * np.pi) / auxiliary_var)  # Compute omega factor for current k.
        
            for i in range(k, n, 2 ** (m + 1)):  # For all values of i, j with i = [k,k+2^(m+1),k+2*2^(m+1),...,n[
                # Perform elementary transformation.
                # j = i+2^m
                p = omega * fdata[i + 2**m]  # p = f[j]*e^((-2*pi*k*i)/(2^(m+1)))
                fdata[i + 2**m] = fdata[i] - p  # f[j] = f[i]-p
                fdata[i] = fdata[i] + p  # f[i] = f[i]+p
                
        auxiliary_var *= 2
        m += 1
    
    # TODO: normalize fft signal
    fdata = np.dot(1 / np.sqrt(n), fdata)

    return fdata

# http://www.informatik.tu-freiberg.de/lehre/pflicht/EinInf/ws07/Informatik10.pdf
def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)
    
    x_max = 2 * np.pi
    
    evenly_spaced_num = np.linspace(x_min, x_max, num_samples)
    
    # TODO: Generate sine wave with proper frequency
    data = np.sin(f * evenly_spaced_num)

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # TODO: compute Fourier transform of input data
    FT_adata = np.fft.fft(adata)
    print(FT_adata)
    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    n = adata.size
    """
    Set values above bandlimit and (n - bandlimit_index) to zero.
    Here you can see the almost symmetry (the elements are repeated):
    [ -4.741+17.901j -28.856 -2.089j  18.604-30.368j ...  11.292 -8.425j  18.604+30.368j -28.856 +2.089j]
    """
    FT_adata[bandlimit_index + 1: n - bandlimit_index] = 0

    # TODO: compute inverse transform and extract real component
    adata_filtered = np.zeros(adata.shape[0])
    inv = np.fft.ifft(FT_adata)  # np.fft.ifft computes the one-dimensional inverse discrete FT.
    adata_filtered = np.real(inv)  # np.real teturns the real part of the complex argument.

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
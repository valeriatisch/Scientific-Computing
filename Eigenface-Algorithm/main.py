import numpy as np
import lib
import matplotlib as mpl


####################################################################################################
# Exercise 1: Power Iteration http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter7.pdf

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    # TODO: set epsilon to default value if not set by user
    epsilon = np.finfo(M.dtype).eps
    rows, columns = M.shape

    # TODO: random vector of proper size to initialize iteration
    vector = np.random.randn(rows)

    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon

    # Perform power iteration
    while residual > epsilon:
        # TODO: implement power iteration
        eigenvector = np.dot(M, vector)  # Calculates the matrix vector product.
        eigenvector_norm = np.linalg.norm(eigenvector)  # Calculates the norm/length of ev.
        eigenvector = eigenvector / eigenvector_norm  # Renormalizes the ev.
        rest = eigenvector - vector  # Calculates the residual.
        vector = eigenvector  # next ev
        residual = np.linalg.norm(rest)
        residuals.append(residual)
        pass
                
    return vector, residuals

####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """
    
    images = []

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()
    
    file_list = lib.list_directory(path)  # Returns a list containing the names of the entries in the directory given by path.
    file_list.sort()
    
    for file in file_list:  # Go through each file.
        if file.endswith(".png"):  # Is it a png file?
            an_array = mpl.image.imread(path + file)  # Read an image from a file into an array.
            an_array = np.float64(an_array)  # Convert to float64.
            as_ndarray = np.asarray(an_array)  # out: ndarray
            images.append(as_ndarray)  # Add to the end of the list.
            
    # TODO set dimensions according to first image in images
    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # TODO: initialize data matrix with proper size and data type
    rows = len(images)
    columns = images[0].size
    D = np.zeros((rows, columns))

    # TODO: add flattened images to data matrix
    for row in range(rows):
        flattened_img = np.ndarray.flatten(images[row])  # images flatten
        D[row] = flattened_img  # Each row in D is a flatten image.
         
    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """
    
    # TODO: subtract mean from data / center data at origin
    mean_data = np.zeros((1, 1))
    rows, columns = D.shape
    # D[np.where(D == 0)] = np.nan
    mean_data = np.mean(D, axis = 0)  # compute the mean of each column 
    
    for c in range(rows):  # subtract computed mean from each row / normalize the matrix
        D[c] -= mean_data
    
    # TODO: compute left and right singular vectors and singular values
    # Useful functions: numpy.linalg.svd(..., full_matrices=False)
    svals, pcs = [np.ones((1, 1))] * 2
    
    U, svals, pcs = np.linalg.svd(D, full_matrices = False)

    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    # TODO: Normalize singular value magnitudes
    elements = singular_values.shape[0]
    sv_sum = np.sum(singular_values)
    
    for element in range(elements):
        singular_values[element] /= sv_sum
    
    k = 0
    # TODO: Determine k that first k singular values make up threshold percent of magnitude
    element_sum = 0
    
    for element in range(elements):  # Go through each element in sv-vector.
        if element_sum < threshold:  # How many k sv-vectors do I need to get 80% of all sv-vectors?
            k = k + 1  # the index of element_sum
            element_sum += singular_values[element]  # Sums the current element and last element.
        else:
            pass
    
    lib.plot_singular_values_and_energy(singular_values, k)
    
    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # TODO: initialize coefficients array with proper size
    rows = len(images)  # Each row contains coefficients of one image. -> number of rows of coeff = number of images  
    columns = pcs.shape[0]  # psc = basis = V = matrix containing principal components as row -> psc's number of rows = number of columns
    coefficients = np.zeros((rows, columns))

    # TODO: iterate over images and project each normalized image into principal component basis
    flatten_img = setup_data_matrix(images)
    mittelwertbefreites_img = flatten_img - mean_data  # Subtracta mean from data matrix (normalize).
    
    for i in range(rows): 
        coefficients[i] = np.dot(pcs, mittelwertbefreites_img[i])  # Projects the images into subspace to compare easier.
    return coefficients


# https://stackoverflow.com/questions/39497496/angle-between-two-vectors-3d-python
def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    # TODO: load test data set
    imgs_test = []
    file_ending = ".png"
    imgs_test, dimension_y, dimension_x = load_images(path_test, file_ending)

    # TODO: project test data set into eigenbasis
    coeffs_test = np.zeros(coeffs_train.shape)
    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    # TODO: Initialize scores matrix with proper size
    rows = coeffs_train.shape[0]
    columns = len(imgs_test)
    scores = np.zeros((rows, columns))

    # TODO: Iterate over all images and calculate pairwise correlation
    for i in range(columns):
        for j in range(rows):
            dot_product = np.dot(coeffs_test[i], coeffs_train[j])
            length = np.linalg.norm(coeffs_test[i]) * np.linalg.norm(coeffs_train[j])
            angle = np.arccos(dot_product / length)  # Calculates the angle between two vectors.
            scores[j, i] = angle 
            
    return scores, imgs_test, coeffs_test


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")

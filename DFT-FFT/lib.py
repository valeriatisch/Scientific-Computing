
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavio

from main import dft_matrix, is_unitary, fft

'''
    This package is to be used as a library. Please do not edit.
'''
# convenience adjustments
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)


def dft(data: np.ndarray, test: bool = False) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data by constructing DFT matrix.

    Arguments:
    data: data to be transformed (np.array, shape=(n,), dtype='float64')
    test: if true the correctness of the transform is verified using suitable test cases

    Return:
    fdata: Fourier transformed data
    """
    fdata = data.copy()
    # compute DFT matrix and check if unitary
    F = dft_matrix(fdata.size)
    if test and not is_unitary(F):
        raise ValueError("Cannot calculate DFT")

    # perform Fourier transform
    fdata = F.dot(fdata)

    return fdata


def idft(data: np.ndarray) -> np.ndarray:
    """
    Perform inverse discrete Fourier transform of data by conjugating signal.

    Arguments:
    data: frequency data to be transformed (np.array, shape=(n,), dtype='float64')

    Return:
    result: Inverse transformed data
    """
    n = len(data)
    result = np.conjugate(dft(np.conjugate(data)))
    return result


def plot_harmonics(sigs: list, fsigs: list):
    """
    Plots the signals and its fourier transforms in two columns

    Arguments:
    sigs: the signal list
    fsigs: the fourier transformations of the signals
    """

    # plot the first 10 harmonic components
    n_plots = 10
    fig = plt.figure(figsize=(15, 8))
    for i in range(n_plots):

        fig.add_subplot(n_plots, 2, 2 * i + 1)
        plt.stem(sigs[i], '-rx')
        plt.xlim(0, 128)
        plt.yticks([])
        if i < n_plots - 1:
            plt.xticks([])

        fig.add_subplot(n_plots, 2, 2 * i + 2)
        plt.plot(np.real(fsigs[i]))
        plt.plot(np.imag(fsigs[i]))
        plt.xlim(0, 128)
        plt.yticks([])
        if i < n_plots - 1:
            plt.xticks([])

    plt.show()


def ifft(data: np.ndarray) -> np.ndarray:
    """
    Perform inverse discrete Fast Fourier transform of data by conjugating signal.

    Arguments:
    data: frequency data to be transformed (np.array, shape=(n,), dtype='float64')

    Return:
    result: Inverse transformed data
    """
    n = len(data)
    result = np.conjugate(fft(np.conjugate(data)))
    return result


def read_audio_data(fname: str) -> tuple:
    """
    Read audio data from file and return numpy array representation.

    Arguments:
    fname: filename of audio file

    Return:
    adata: audio data as numpy ndarray (shape=(n,), dtype=float64)
    rate: audio parameters (useful for generating output matching input)
    """

    (rate, adata_uint) = wavio.read(fname)

    # cast to float64 to perform subsequent computation in convenient
    # floating point format
    adata = np.asarray(adata_uint, dtype='float64')
    # for symmetry with writeAudioData(); scaling in geneal unclear
    adata /= (2 ** 15 - 1)

    return adata, rate


def write_audio_data(fname: str, data: np.ndarray, rate: int):
    """
    Write audio data given as numpy array to fname in WAV format

    Arguments:
    fname: name of WAV audio file to be written.
    data: audio data to be written (shape=(n,), dtype=float64)
    rate: sampling rate per second

    Side effects:
    Creates WAV file fname.
    """

#    scaled_data = np.int16(data / np.max(np.abs(data)) * (2 ** 15 - 1))
    wavio.write(fname, rate, data)


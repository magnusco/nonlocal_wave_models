import numpy as np
from scipy import fft, optimize
import matplotlib.pyplot as plt


def func_guess(X, height, P):
    return height * np.cos(((2 * np.pi) / P) * X)


def bessel_symbol(Xi, s):
    return np.power((1 + np.power(Xi, 2)), -s / 2)


def FDP_const_sol(wavespeed, integration_const):
    return (wavespeed + np.sqrt(wavespeed ** 2 + 8 * integration_const)) / 4


def FKDV_system(sol, height, P, s, symbol_values):
    phi, wavespeed = sol[:-1], sol[-1]
    phi_coeffs = fft.rfft(phi)
    scaled_phi_coeffs = fft.ifftshift(
        np.multiply(fft.fftshift(phi_coeffs), symbol_values)
    )

    equation_error = (
        wavespeed * phi - (1 / 2) * np.power(phi, 2) - fft.irfft(scaled_phi_coeffs)
    )
    height_error = np.array([np.abs(height - phi[int(len(phi) / 2) + 1])])
    return np.concatenate((equation_error, height_error))


if __name__ == "__main__":
    s = 0.5
    P = 2 * np.pi
    N = 2 ** 8
    X = np.linspace(-P / 2, P / 2, N, endpoint=False)

    bifurcation_point = bessel_symbol((2 * np.pi) / P, s)
    symbol_values = bessel_symbol(
        np.array([(2 * np.pi * (k - N / 2)) / P for k in range(0, N)]), s
    )
    H = np.linspace(0, 0.3 * bifurcation_point, 30)
    wavespeeds, max_heights = np.zeros(len(H)), np.zeros(len(H))

    for i in range(0, len(H)):
        initial_phi_guess = func_guess(X, H[i], P)
        initial_wavespeed_guess = np.array([bifurcation_point])

        F = lambda sol: FKDV_system(sol, H[i], P, s, symbol_values)
        solution = optimize.fsolve(
            F, np.concatenate((initial_phi_guess, initial_wavespeed_guess))
        )

        wavespeeds[i] = solution[-1]
        max_heights[i] = np.max(solution[:-1])
        plt.plot(X, solution[:-1], linewidth=0.7)
    plt.show()
    plt.plot(wavespeeds, max_heights)
    plt.show()

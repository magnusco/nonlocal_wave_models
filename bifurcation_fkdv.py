import numpy as np
from scipy import fft, optimize
import matplotlib.pyplot as plt


def func_guess(X_half, height, P):
    return height * np.cos(((2 * np.pi) / P) * X_half)


def bessel_symbol(Xi, s):
    return (1 + Xi ** 2) ** (-s / 2)


def even_half_to_full_P(half):
    return np.concatenate((half, np.flip(half[1:-1])))


def odd_half_to_full_P(half):
    return np.concatenate((half, -np.flip(half[1:-1])))


def FKDV_system(sol, height, P, s, symb_coeffs):
    phi_half, mu = sol[:-1], sol[-1]
    phi = np.concatenate((phi_half, np.flip(phi_half[1:-1])))

    phi_coeffs = fft.dct(phi_half, type=1)
    scaled_phi_coeffs = np.multiply(phi_coeffs, 2 * symb_coeffs)
    smoothed_phi_half = fft.idct(scaled_phi_coeffs, type=1)
    smoothed_phi = np.concatenate((smoothed_phi_half, np.flip(smoothed_phi_half[1:-1])))

    eq_err = (mu * phi) - ((1 / 2) * phi ** 2) - smoothed_phi
    height_err = np.array([np.abs(height - phi[int(len(phi) / 2) + 1])])
    return np.concatenate((eq_err[: len(sol) - 1], height_err))


if __name__ == "__main__":
    s = 0.5
    P = 2 * np.pi
    N = 2 ** 7
    X_half = np.linspace(-P / 2, 0, N)
    samples = 10

    bif_point = bessel_symbol((2 * np.pi) / P, s)
    symb_coeffs = bessel_symbol(np.array([(2 * np.pi * k) / P for k in range(0, N)]), s)
    H = np.linspace(0, 0.2 * bif_point, samples)
    wavespeeds, max_heights = np.zeros(samples), np.zeros(samples)

    for i in range(0, samples):
        phi_guess = func_guess(X_half, H[i], P)
        wavespeed_guess = np.array([0.5 * bif_point])

        F = lambda sol: FKDV_system(sol, H[i], P, s, symb_coeffs)
        solution = optimize.fsolve(F, np.concatenate((phi_guess, wavespeed_guess)))

        wavespeeds[i] = solution[-1]
        max_heights[i] = np.max(solution[:-1])
        plt.plot(X_half, solution[:-1], "k", linewidth=0.7)
    plt.show()
    plt.plot(wavespeeds, max_heights, "k")
    plt.show()

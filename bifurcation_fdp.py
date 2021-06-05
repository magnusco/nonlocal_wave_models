import numpy as np
from scipy import fft, optimize
import matplotlib.pyplot as plt


def func_guess(X, height, P):
    return height * np.cos(((2 * np.pi) / P) * X)


def bessel_symbol(Xi, s):
    return np.power((1 + np.power(Xi, 2)), -s / 2)


def FDP_const_sol(wavespeed, integration_const):
    return (wavespeed + np.sqrt(wavespeed ** 2 + 8 * integration_const)) / 4


def FDP_system(
    sol, height, P, s, symbol_values, integration_const,
):
    phi, wavespeed = sol[:-1], sol[-1]
    squared_phi_coeffs = fft.dct(np.power(phi, 2), type=1)
    scaled_phi_coeffs = np.multiply(squared_phi_coeffs, symbol_values)

    equation_error = (
        wavespeed * phi
        - (1 / 2) * np.power(phi, 2)
        - (3 / 2) * fft.idct(scaled_phi_coeffs, type=1)
        - integration_const
    )
    height_error = np.array(np.abs(height - phi[int(len(phi) / 2) + 1]))
    return np.concatenate((equation_error, height_error))


if __name__ == "__main__":
    s = 0.5
    P = 0.9 * ((2 * np.pi) / np.sqrt(3 ** (2 / s) - 1))
    N = 2 ** 8
    X = np.linspace(-P / 2, P / 2, N, endpoint=False)
    integration_const = 1

    implicit_bifurcation_point = lambda wavespeed: 3 * bessel_symbol(
        (2 * np.pi) / P, s
    ) - (wavespeed - FDP_const_sol(wavespeed, integration_const)) / FDP_const_sol(
        wavespeed, integration_const
    )
    bifurcation_point = optimize.fsolve(implicit_bifurcation_point, 1)

    symbol_values = bessel_symbol(
        np.array([(2 * np.pi * k) / P for k in range(0, len(X))]), s
    )
    H_relative = np.linspace(
        0,
        0.1 * (bifurcation_point - FDP_const_sol(bifurcation_point, integration_const)),
        10,
    )

    for i in range(0, len(H_relative)):
        initial_phi_guess = func_guess(X, H_relative[i], P) + FDP_const_sol(
            bifurcation_point, integration_const
        )
        initial_wavespeed_guess = np.array(bifurcation_point)

        F = lambda sol: FDP_system(
            sol, H_relative[i], P, s, symbol_values, integration_const
        )
        solution = optimize.fsolve(
            F, np.concatenate((initial_phi_guess, initial_wavespeed_guess))
        )
        print(solution[-1])
        plt.plot(X, solution[:-1], linewidth=0.7)
    plt.show()

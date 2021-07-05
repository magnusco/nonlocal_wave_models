import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def func_guess(X, height, P):
    return height * np.cos(((2 * np.pi) / P) * X)


def bessel_symbol(Xi, s):
    return (1 + Xi ** 2) ** (-s / 2)


def homog_symbol(Xi, r):
    return (1j * Xi) ** r


def const_sol(mu, int_const):
    return (mu + np.sqrt(mu ** 2 + 6 * int_const)) / 3


def upper_bound_P(s):
    return (2 * np.pi) / np.sqrt(4 ** (2 / s) - 1)


def DP_CH_system(sol, height, P, s, r, symb_coeffs, homog_symb_coeffs, int_const):
    phi, mu = sol[:-1], sol[-1]
    gamma = const_sol(mu, int_const)
    phi_coeffs = np.fft.rfft(phi)
    phi_squared_coeffs = np.fft.rfft(phi ** 2)
    mod_phi_coeffs = np.multiply(phi_coeffs, symb_coeffs)
    mod_phi_squared_coeffs = np.multiply(phi_squared_coeffs, symb_coeffs)
    frac_der_phi = np.fft.irfft(np.multiply(phi_coeffs, homog_symb_coeffs))
    frac_der_phi_squared_coeffs = np.fft.rfft(frac_der_phi ** 2)

    eq_err = (
        (gamma - mu) * phi
        - (1 / 2) * phi ** 2
        - np.fft.irfft(mod_phi_squared_coeffs)
        + 2 * gamma * np.fft.irfft(mod_phi_coeffs)
        - (1 / 2) * np.fft.irfft(np.multiply(symb_coeffs, frac_der_phi_squared_coeffs))
    )
    height_err = np.array([np.abs(height - phi[(len(phi) // 2) + 1])])
    return np.concatenate((eq_err, height_err))


if __name__ == "__main__":
    s = 2
    r = 0.1
    P = 0.5 * upper_bound_P(s)
    N = 500
    X = np.linspace(-P / 2, P / 2, N, endpoint=False)
    int_const = 1
    samples = 10

    impl_bif_point = lambda mu: (
        3 * const_sol(mu, int_const) * bessel_symbol((2 * np.pi) / P, s)
    ) - (mu - const_sol(mu, int_const))

    bif_point = optimize.fsolve(impl_bif_point, 1)[0]
    frequencies = np.array([(2 * np.pi * k) / P for k in range(0, (N // 2) + 1)])
    symb_coeffs = bessel_symbol(frequencies, s)
    homog_symb_coeffs = homog_symbol(frequencies, r)

    min_height = const_sol(bif_point, int_const) - bif_point
    H = np.linspace(0, 0.5 * min_height, samples)
    wavespeeds, max_heights = np.zeros(samples), np.zeros(samples)
    wavespeed_guess = np.array([bif_point])

    for i in range(0, samples):
        phi_guess = func_guess(X, H[i], P)
        F = lambda sol: DP_CH_system(
            sol, H[i], P, s, r, symb_coeffs, homog_symb_coeffs, int_const
        )
        solution = optimize.fsolve(
            F, np.concatenate((phi_guess, wavespeed_guess)), xtol=1e-5
        )
        varphi = const_sol(solution[-1], int_const) - solution[:-1]
        wavespeeds[i] = solution[-1]
        max_heights[i] = np.max(varphi)
        wavespeed_guess = np.array([solution[-1]]) + 0.5
        plt.plot(X, varphi, "k", linewidth=0.7)

    plt.ylabel(r"$\varphi (x)$")
    plt.xlabel(r"$x$")
    plt.show()

    plt.plot(wavespeeds, max_heights, "k", linewidth=0.4)
    plt.plot(wavespeeds, max_heights, "k.", linewidth=0.7)
    plt.plot(wavespeeds, wavespeeds, "r")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\max \ \varphi$")
    plt.plot(wavespeeds, const_sol(wavespeeds, int_const), "b")
    plt.show()

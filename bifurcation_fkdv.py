import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def func_guess(X, height, P):
    return height * np.cos(((2 * np.pi) / P) * X)


def bessel_symbol(Xi, s):
    return (1 + Xi ** 2) ** (-s / 2)


def FKDV_system(sol, height, P, s, symb_coeffs):
    phi, mu = sol[:-1], sol[-1]

    phi_coeffs = np.fft.rfft(phi)
    mod_phi_coeffs = np.multiply(phi_coeffs, symb_coeffs)

    eq_err = (mu * phi) - ((1 / 2) * phi ** 2) - np.fft.irfft(mod_phi_coeffs)
    height_err = np.array([np.abs(height - phi[(len(phi) // 2) + 1])])
    return np.concatenate((eq_err, height_err))


if __name__ == "__main__":
    s = 0.5
    P = 2 * np.pi
    N = 1500
    X = np.linspace(-P / 2, P / 2, N, endpoint=False)
    samples = 14

    bif_point = bessel_symbol((2 * np.pi) / P, s)
    symb_coeffs = bessel_symbol(
        np.array([(2 * np.pi * k) / P for k in range(0, (N // 2) + 1)]), s
    )
    H = np.linspace(0, 0.7 * bif_point, samples)
    wavespeeds, max_heights = np.zeros(samples), np.zeros(samples)

    for i in range(0, samples):
        wavespeed_guess = np.array([bif_point])
        phi_guess = func_guess(X, H[i], P)

        F = lambda sol: FKDV_system(sol, H[i], P, s, symb_coeffs)
        solution = optimize.fsolve(
            F, np.concatenate((phi_guess, wavespeed_guess)), xtol=1e-6
        )

        wavespeeds[i] = solution[-1]
        max_heights[i] = np.max(solution[:-1])
        plt.plot(X, solution[:-1], "k", linewidth=0.7)

    plt.ylabel(r"$\varphi (x)$")
    plt.xlabel(r"$x$")
    # plt.savefig("bifurcation_fkdv.png")
    plt.show()
    plt.plot(wavespeeds, max_heights, "k", linewidth=0.4)
    plt.plot(wavespeeds, max_heights, "k.", linewidth=0.7)
    plt.plot(np.linspace(0.73, 0.87, 50), np.linspace(0.73, 0.87, 50), "k")
    plt.plot(np.linspace(0.73, 0.87, 50), np.zeros(50), "b")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\max \ \varphi$")
    # plt.savefig("bifurcation_branch_fkdv.png")
    plt.show()

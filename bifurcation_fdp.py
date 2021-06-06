import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def func_guess(X, height, P):
    return height * np.cos(((2 * np.pi) / P) * X)


def bessel_symbol(Xi, s):
    return (1 + Xi ** 2) ** (-s / 2)


def const_sol(mu, int_const):
    return (mu + np.sqrt(mu ** 2 + 8 * int_const)) / 4


def upper_bound_P(s):
    return (2 * np.pi) / np.sqrt(3 ** (2 / s) - 1)


def FDP_system(sol, height, P, s, symb_coeffs, int_const):
    phi, mu = sol[:-1], sol[-1]

    phi_coeffs = np.fft.rfft(phi ** 2)
    mod_phi_coeffs = np.multiply(phi_coeffs, symb_coeffs)

    eq_err = (
        mu * phi
        - (1 / 2) * phi ** 2
        - (3 / 2) * np.fft.irfft(mod_phi_coeffs)
        - int_const
    )
    height_err = np.array([np.abs(height - phi[(len(phi) // 2) + 1])])
    return np.concatenate((eq_err, height_err))


if __name__ == "__main__":
    s = 0.5
    P = 0.9 * upper_bound_P(s)
    N = 500
    X = np.linspace(-P / 2, P / 2, N, endpoint=False)
    int_const = 1
    samples = 5

    impl_bif_point = lambda mu: (
        3 * const_sol(mu, int_const) * bessel_symbol((2 * np.pi) / P, s)
    ) - (mu - const_sol(mu, int_const))

    bif_point = optimize.fsolve(impl_bif_point, 1)[0]
    symb_coeffs = bessel_symbol(
        np.array([(2 * np.pi * k) / P for k in range(0, (N // 2) + 1)]), s
    )

    H = np.linspace(
        const_sol(bif_point, int_const),
        bif_point - 0.7 * const_sol(bif_point, int_const),
        samples,
    )
    wavespeeds, max_heights = np.zeros(samples), np.zeros(samples)

    print(const_sol(bif_point, int_const))
    print(bif_point - 0.7 * const_sol(bif_point, int_const))

    for i in range(0, samples):
        phi_guess = func_guess(
            X, H[i] - const_sol(bif_point, int_const), P
        ) + const_sol(bif_point, int_const)
        wavespeed_guess = np.array([bif_point])

        F = lambda sol: FDP_system(sol, H[i], P, s, symb_coeffs, int_const)
        solution = optimize.fsolve(
            F, np.concatenate((phi_guess, wavespeed_guess)), xtol=1e-5
        )

        wavespeeds[i] = solution[-1]
        max_heights[i] = np.max(solution[:-1])
        plt.plot(X, solution[:-1], "k", linewidth=0.7)

    plt.show()
    plt.plot(wavespeeds, max_heights, "k")
    plt.show()

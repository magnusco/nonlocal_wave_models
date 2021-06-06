import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def bessel_symbol(Xi, s):
    return (1 + Xi ** 2) ** (-s / 2)


def const_sol(mu, int_const):
    return (mu + np.sqrt(mu ** 2 + 8 * int_const)) / 4


def const_sol_der(mu, int_const):
    return (1 / 4) * (1 + np.divide(mu, np.sqrt(np.power(mu, 2) + 8 * int_const)))


def upper_bound_P(s):
    return (2 * np.pi) / np.sqrt(3 ** (2 / s) - 1)


def mu_2(Xi, s, symbol):
    m_1 = bessel_symbol(Xi, s)
    m_2 = bessel_symbol(2 * Xi, s)
    return (1 / (1 - m_1)) - ((1 + 3 * m_2) / (8 * (m_1 - m_2)))


if __name__ == "__main__":
    s_samples = 20
    dx = 1e-3
    S = np.linspace(0.2, 0.9, s_samples)
    int_const = 1
    s = 0.9

    P = upper_bound_P(s)
    P_values = np.arange(1e-1, P, dx)
    Xi = (2 * np.pi) / P_values

    mu_2_values = mu_2(Xi, s, bessel_symbol)
    plt.plot(P_values, mu_2_values, "k", linewidth=0.7)
    plt.plot(P_values, np.zeros(len(P_values)), "k", linewidth=0.7)
    plt.ylim([-5, 5])
    plt.show()

    exit()

    ############################################################################
    min_mu_2_values = np.zeros(s_samples)
    max_mu_2_values = np.zeros(s_samples)
    for i in range(0, s_samples):
        P = upper_bound_P(S[i])
        Xi = (2 * np.pi) / np.arange(1e-5, P, dx)

        impl_bif_points = lambda mu: (mu - const_sol(mu, int_const)) - (
            3 * const_sol(mu, int_const) * bessel_symbol(Xi, S[i])
        )

        bif_points = fsolve(impl_bif_points, np.ones(len(Xi)))
        const_sols = const_sol(bif_points, int_const)
        const_sol_ders = const_sol_der(bif_points, int_const)

        mu_2_values = mu_2(
            Xi, S[i], bif_points, const_sols, const_sol_ders, bessel_symbol
        )
        min_mu_2_values[i] = np.min(np.abs(mu_2_values))

    plt.xlabel(r"$s$")
    plt.ylabel(r"$\min\ |\mu_2|$")
    plt.plot(S, min_mu_2_values, "k")
    plt.savefig("mu_2_fdp.png")
    plt.show()
    plt.close()

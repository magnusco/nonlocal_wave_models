import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def bessel_symbol(Xi, s):
    return (1 + Xi ** 2) ** (-s / 2)


def upper_bound_P(s):
    return (2 * np.pi) / np.sqrt(3 ** (2 / s) - 1)


def const_sol(mu, int_const):
    return (mu + np.sqrt(mu ** 2 + 8 * int_const)) / 4


def const_sol_der(mu, int_const):
    return (1 + (mu / np.sqrt(mu ** 2 + 8 * int_const))) / 4


def mu_2_simplified(Xi, s):
    m_1 = bessel_symbol(Xi, s)
    m_2 = bessel_symbol(2 * Xi, s)
    return 9 * m_1 + 3 * m_1 * m_2 - 11 * m_2 - 1


def mu_2(Xi, s, int_const, bif_points):
    m_1 = bessel_symbol(Xi, s)
    m_2 = bessel_symbol(2 * Xi, s)
    a = (1 + 3 * m_1) / (3 * const_sol(bif_points, int_const))
    b = 1 / (
        3 * m_1 * const_sol_der(bif_points, int_const)
        + const_sol_der(bif_points, int_const)
        - 1
    )
    c = (1 / (1 - m_1)) - (1 + 3 * m_2) / (8 * (m_1 - m_2))
    return a * b * c


if __name__ == "__main__":

    dx = 1e-4
    S = np.linspace(0.1, 0.9, 9)
    int_const = 1

    for s in S:
        P = upper_bound_P(s)
        P_values = np.arange(1e-4, P, dx)
        Xi = (2 * np.pi) / P_values
        number_periods = len(P_values)

        # impl_bif_points = lambda mu: (mu - const_sol(mu, int_const)) - (
        #     3 * const_sol(mu, int_const) * bessel_symbol(Xi, s)
        # )
        # bif_points = fsolve(impl_bif_points, np.ones(number_periods))

        mu_2_values = mu_2_simplified(Xi, s)
        # mu_2_values = mu_2(Xi, s, int_const, bif_points)

        plt.plot(P_values, mu_2_values, "k", linewidth=0.7)
        plt.plot(P_values, np.zeros(number_periods), "k", linewidth=0.7)

    plt.xlabel(r"$P$")
    plt.ylabel(r"$\mu_2$")
    plt.ylim([-1.25, 0.5])
    plt.text(-0.01, -0.65, r"$s = $" + str(0.1))
    plt.text(0.9, -0.17, r"$s = $" + str(0.6))
    plt.text(1.8, 0.2, r"$s = $" + str(0.9))
    plt.text(0.2, 0.02, r"$\mu_2 = 0$")
    plt.savefig("mu_2_fdp.png")
    plt.show()
    plt.close()

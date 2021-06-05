import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def upper_bound_P(s):
    return (2 * np.pi) / np.sqrt(np.power(3, 2 / s) - 1)


def const_sol(wavespeed, integration_const):
    return (wavespeed + np.sqrt(np.power(wavespeed, 2) + 8 * integration_const)) / 4


def const_sol_der(wavespeed, integration_const):
    return (1 / 4) * (
        1
        + np.divide(wavespeed, np.sqrt(np.power(wavespeed, 2) + 8 * integration_const))
    )


def bessel_symbol(Xi, s):
    return np.power((1 + np.power(Xi, 2)), -s / 2)


def mu_2(Xi, s, bifurcation_point, const_sol, const_sol_der, symbol, integration_const):
    a = 1 + 3 * symbol(Xi, s)
    b = (3 * symbol(Xi, s) + 1) * const_sol_der(
        bifurcation_point, integration_const
    ) - 1
    c = 1 / (4 * (const_sol(bifurcation_point, integration_const) - bifurcation_point))

    d = 8 * (
        3 * const_sol_der(bifurcation_point, integration_const) * symbol(2 * Xi, s)
        - bifurcation_point
        + const_sol(bifurcation_point, integration_const)
    )

    return (a / b) * (c + (a / d))


def mu_2_part(Xi, s, symbol):
    return 8 * (symbol(Xi, s) - symbol(2 * Xi, s)) - (
        (1 - symbol(Xi, s)) * (1 + 3 * symbol(2 * Xi, s))
    )


if __name__ == "__main__":
    N = 15
    M = int(1e4)
    S = np.linspace(0.3, 2, N)
    integration_const = 1
    smallest_mu_2_values = np.zeros(N)
    largest_mu_2_values = np.zeros(N)

    for i in range(0, N):
        P = upper_bound_P(S[i])
        Xi = (2 * np.pi) / np.linspace(1e-6, P, M, endpoint=False)
        symbol = bessel_symbol

        # bifurcation_points_implicit = lambda wavespeed: (
        #     wavespeed - const_sol(wavespeed, integration_const)
        # ) - 3 * const_sol(wavespeed, integration_const) * bessel_symbol(Xi, S[i])

        # bifurcation_points = fsolve(bifurcation_points_implicit, np.ones(M))

        mu_2_values = mu_2_part(Xi, S[i], symbol)
        smallest_mu_2_values[i] = np.min(np.abs(mu_2_values))

    plt.xlabel(r"$s$")
    plt.ylabel(r"$\min\ |\mu_2|$")
    plt.plot(S, smallest_mu_2_values, "k")
    plt.savefig("mu_2_fdp.png")
    plt.show()
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def bessel_symbol(Xi, s):
    return np.power((1 + np.power(Xi, 2)), -s / 2)


def whitham_symbol(Xi, s):
    return np.power(np.divide(np.tanh(Xi), Xi), s)


def mu_2(Xi, s, symbol):
    return 1 / (4 * (symbol(Xi, s) - 1)) + 1 / (8 * (symbol(Xi, s) - symbol(2 * Xi, s)))


def mu_4(Xi, s, symbol):
    q = (
        1
        / 32
        * (
            1 / np.power(1 - symbol(Xi, s), 3)
            - 1 / (np.power(1 - symbol(Xi, s), 2) * (symbol(Xi, s) - symbol(2 * Xi, s)))
        )
    )
    r = (
        1
        / 64
        * (
            -1 / ((1 - symbol(Xi, s)) * np.power(symbol(Xi, s) - symbol(2 * Xi, s), 2))
            - 1 / np.power(symbol(Xi, s) - symbol(2 * Xi, s), 3)
            - 3
            / (
                np.power(symbol(Xi, s) - symbol(2 * Xi, s), 2)
                * (symbol(3 * Xi, s) - symbol(1 * Xi, s))
            )
        )
    )
    return q + r


if __name__ == "__main__":
    P = np.linspace(0.001, 10, 1000)
    Xi = (2 * np.pi) / P
    symbol = bessel_symbol

    S = np.linspace(0.1, 0.9, 9)
    plt.plot(P, np.zeros(len(P)), "k", linewidth=0.7)
    for s in S:
        Mu_2 = mu_2(Xi, s, symbol)
        plt.plot(P, Mu_2, "k", linewidth=0.7)
    plt.ylim([-10, 5])
    plt.text(8.5, -7.7, r"$s = $" + str(0.1))
    plt.text(8.5, -0.7, r"$s = $" + str(0.9))
    plt.xlabel(r"$P$")
    plt.ylabel(r"$\mu_2(P)$")
    plt.savefig("mu_2_fkdv.png")
    plt.show()
    plt.close()

    S = np.linspace(0.01, 0.99, 1000)
    mu_4_values = np.zeros(len(S))
    for i in range(0, len(S)):
        mu_2_partial = lambda P: mu_2((2 * np.pi) / P, S[i], symbol)
        illegal_P = fsolve(mu_2_partial, 1)
        mu_4_values[i] = mu_4((2 * np.pi) / illegal_P, s, bessel_symbol)
    plt.plot(S, mu_4_values, "k", linewidth=0.9)
    plt.plot(S[-1], mu_4_values[-1], "ro")
    plt.text(
        0.79, 0.97, r"$\min\ \mu_4 \approx $" + str(round(np.amin(mu_4_values), 3))
    )
    plt.xlabel(r"$s$")
    plt.ylabel(r"$\mu_4(P^*_s)$")
    plt.savefig("mu_4_fkdv.png")
    plt.show()
    plt.close()

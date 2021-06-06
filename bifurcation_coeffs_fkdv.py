import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def bessel_symbol(Xi, s):
    return (1 + Xi ** 2) ** (-s / 2)


def mu_2(Xi, s, symbol):
    return 1 / (4 * (symbol(Xi, s) - 1)) + 1 / (8 * (symbol(Xi, s) - symbol(2 * Xi, s)))


def mu_4(Xi, s, symbol):
    q = (
        1
        / 32
        * (
            1 / (1 - symbol(Xi, s) ** 3)
            - 1 / ((1 - symbol(Xi, s) ** 2) * (symbol(Xi, s) - symbol(2 * Xi, s)))
        )
    )
    r = (
        1
        / 64
        * (
            -1 / ((1 - symbol(Xi, s)) * (symbol(Xi, s) - symbol(2 * Xi, s) ** 2))
            - 1 / (symbol(Xi, s) - symbol(2 * Xi, s) ** 3)
            - 3
            / (
                (symbol(Xi, s) - symbol(2 * Xi, s) ** 2)
                * (symbol(3 * Xi, s) - symbol(1 * Xi, s))
            )
        )
    )
    return q + r


if __name__ == "__main__":
    N = 1000
    P = np.linspace(0.001, 10, N)
    Xi = (2 * np.pi) / P

    S = np.linspace(0.1, 0.9, 9)
    plt.plot(P, np.zeros(N), "k", linewidth=0.7)
    for s in S:
        Mu_2 = mu_2(Xi, s, bessel_symbol)
        plt.plot(P, Mu_2, "k", linewidth=0.7)
    plt.ylim([-10, 5])
    plt.text(8.5, -7.7, r"$s = $" + str(0.1))
    plt.text(8.5, -0.7, r"$s = $" + str(0.9))
    plt.xlabel(r"$P$")
    plt.ylabel(r"$\mu_2$")
    plt.savefig("mu_2_fkdv.png")
    plt.show()
    plt.close()

    M = 1000
    S = np.linspace(1 / M, 1, M, endpoint=False)
    mu_4_values = np.zeros(M)
    for i in range(0, M):
        mu_2_partial = lambda P: mu_2((2 * np.pi) / P, S[i], bessel_symbol)
        illegal_P = fsolve(mu_2_partial, 1)
        mu_4_values[i] = mu_4((2 * np.pi) / illegal_P, s, bessel_symbol)
    plt.plot(S, mu_4_values, "k", linewidth=0.7)
    plt.plot(S[-1], mu_4_values[-1], "ro")
    plt.text(0.85, 0.1, r"$\min\ \mu_4 > 0$")
    print(np.min(mu_4_values))
    plt.xlabel(r"$s$")
    plt.ylabel(r"$\mu_4$")
    plt.savefig("mu_4_fkdv.png")
    plt.show()
    plt.close()

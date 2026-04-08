import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func1(x, a, b, c, d):
    return (a * np.cos(d * x + c)) / x + b

def func2(x, a, b, c):
    return (a * np.cos(x + c)) / x**2 + b


if __name__ == "__main__":

    with open("distanze_massimi.txt") as dist_input:
        distanze = np.array([float(x) for x in dist_input.readlines()])

    with open("multimetro_massimi.txt") as multi_input:
        multi = np.array([float(x) for x in multi_input.readlines()])

    with open("errori_multi_massimi.txt") as err_input:
        err_multi = np.array([float(x) for x in err_input.readlines()])


    # =========================
    # FIT 1/r
    # =========================

    p0_1 = [1., 0., 0., 2 * np.pi / 0.8]

    popt1, pcov1 = curve_fit(
        func1,
        distanze,
        multi,
        sigma=err_multi,
        absolute_sigma=True,
        p0=p0_1
    )

    a1, b1, c1, d1 = popt1
    err1 = np.sqrt(np.diag(pcov1))

    # Chi2
    residuals1 = (multi - func1(distanze, *popt1)) / err_multi
    chi2_1 = np.sum(residuals1**2)
    ndof1 = len(multi) - len(popt1)
    chi2rid_1 = chi2_1 / ndof1

    print("1/r:")
    print(f"a = {a1:.3f} ± {err1[0]:.3f}")
    print(f"b = {b1:.3f} ± {err1[1]:.3f}")
    print(f"c = {c1:.3f} ± {err1[2]:.3f}")
    print(f"d = {d1:.3f} ± {err1[3]:.3f}")
    print("Chi2 ridotto =", chi2rid_1)


    # =========================
    # FIT 1/r^2
    # =========================

    p0_2 = [1., 0., 0.]

    popt2, pcov2 = curve_fit(
        func2,
        distanze,
        multi,
        sigma=err_multi,
        absolute_sigma=True,
        p0=p0_2
    )

    a2, b2, c2 = popt2
    err2 = np.sqrt(np.diag(pcov2))

    residuals2 = (multi - func2(distanze, *popt2)) / err_multi
    chi2_2 = np.sum(residuals2**2)
    ndof2 = len(multi) - len(popt2)
    chi2rid_2 = chi2_2 / ndof2

    print("\n1/r^2:")
    print(f"a = {a2:.3f} ± {err2[0]:.3f}")
    print(f"b = {b2:.3f} ± {err2[1]:.3f}")
    print(f"c = {c2:.3f} ± {err2[2]:.3f}")
    print("Chi2 ridotto =", chi2rid_2)


    # =========================
    # PLOT
    # =========================

    fig, ax = plt.subplots()

    ax.set_title("Andamento $I(r)$")
    ax.set_xlabel("distanza $r$ (cm)")
    ax.set_ylabel("differenza di potenziale $V$ (V)")

    x_axis = np.linspace(min(distanze), max(distanze), 200)

    ax.errorbar(
        distanze,
        multi,
        yerr=err_multi,
        linestyle="None",
        marker="o",
        color="rebeccapurple",
        label="Dati osservati",
        capsize=4
    )

    ax.plot(
        x_axis,
        func1(x_axis, *popt1),
        label="$I \\propto cos(r) / r$",
        color="steelblue"
    )

    ax.plot(
        x_axis,
        func2(x_axis, *popt2),
        label="$I \\propto cos(r) / r^2$",
        color="darkorange"
    )

    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

def inversa1 (x, a, b):
    return a * (1/x) + b

def inversa2 (x, a, b):
    return a * (1/x**2) + b

def lineare (x, m, q):
    return m * x + q

def parabola (x, a, b, c):
    return a * x**2 + b * x + c



if __name__ == "__main__":

    with open ("distanze.txt") as dist_input:
        distanze = [float (x) for x in dist_input.readlines ()]

    with open ("multi.txt") as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open ("err_multi.txt") as err_input:
        err_multi = [float (x) for x in err_input.readlines ()]

    # Calcolo per 1/r

    ls1 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        inversa1
                        )

    myminuit1 = Minuit (ls1,
                        a = 0.,
                        b = 0.
                        )

    myminuit1.migrad ()
    myminuit1.hesse ()

    a1 = myminuit1.values["a"]
    b1 = myminuit1.values["b"]
    a_err1 = myminuit1.errors["a"]
    b_err1 = myminuit1.errors["b"]

    chi2_1 = myminuit1.fval
    ndof_1 = len (distanze) - myminuit1.nfit
    chi2rid_1 = chi2_1 / ndof_1

    print (f"chi2 = {chi2_1}\nchi2 ridotto = {chi2rid_1}")

    # Calcolo per 1/r^2

    ls2 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        inversa2
                        )

    myminuit2 = Minuit (ls2,
                        a = 0.,
                        b = 0.
                        )
    myminuit2.migrad ()
    myminuit2.hesse ()

    a2 = myminuit2.values["a"]
    b2 = myminuit2.values["b"]
    a_err2 = myminuit2.errors["a"]
    b_err2 = myminuit2.errors["b"]

    chi2_2 = myminuit2.fval
    ndof_2 = len (distanze) - myminuit2.nfit
    chi2rid_2 = chi2_2 / ndof_2

    print (f"chi2 = {chi2_2}\nchi2 ridotto = {chi2rid_2}")

    # Fit con lineare

    ls3 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        lineare
                        )

    m3 = Minuit (ls3,
                 m = -0.02,
                 q = 3
                 )

    m3.migrad ()
    m3.hesse ()

    m = m3.values["m"]
    q = m3.values["q"]


    # Parabola

    ls4 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        parabola
                        )

    m4 = Minuit (ls4,
                 a = 300,
                 b = 2.58,
                 c = 0
                 )

    m4.migrad ()
    m4.hesse ()

    a4 = m4.values["a"]
    b4 = m4.values["b"]
    c4 = m4.values["c"]


    # Plot punti

    fig, ax = plt.subplots ()

    ax.set_title ("Andamento generale di $S(r)$")
    ax.set_xlabel ("distanze $r$ (cm)")
    ax.set_ylabel ("differenza di potenziale $\\Delta V$ (V)")

    ax.errorbar (distanze,
                 multi,
                 yerr = err_multi,
                 linestyle = "None",
                 marker = "o",
                 capsize = 4,
                 color = "navy",
                 label = "Dati osservati"
                 )

    # Plot inverse

    x_axis = np.linspace (min(distanze), max(distanze), 200)

    ax.plot (x_axis,
             [inversa1 (x, a1, b1) for x in x_axis],
             label = "$S \\propto 1/r$",
             color = "slateblue"
             )

    ax.plot (x_axis,
             [inversa2 (x, a2, b2) for x in x_axis],
             label = "$S \\propto 1/r^2$",
             color = "deeppink"
             )

    ax.plot (x_axis,
             [lineare (x, m, q) for x in x_axis],
             label = "$S = m x + q$",
             color = "firebrick"
             )

    ax.plot (x_axis,
             [parabola (x, a4, b4, c4) for x in x_axis],
             label = "$S = a x^2 + b + c$",
             color = "purple"
             )

    plt.legend ()
    plt.show ()

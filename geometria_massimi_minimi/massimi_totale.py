import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from pathlib import Path

def func1 (x, a, b, c, d, k):
    return k / x + a * np.cos(d * x + c) + b

def func1_quadro (x, a, b, c, d, k):
    return k / x**2 + a * np.cos(d * x + c) + b


def func2 (x, a, b, c, d, e):
    return a * x**2 + b * x + c * np.cos(d * x) + e

def func3 (x, m, q, A, d, c):
    return m * x + q + A * np.cos(d * x + c)


if __name__ == "__main__":

    current_dir = Path (__file__).parent
    dist_file = current_dir / "distanze_massimi_tot.txt"
    multi_file = current_dir / "multimetro_massimi_tot.txt"
    err_file = current_dir / "errori_multi_massimi_tot.txt"

    with open (dist_file) as dist_input:
        distanze = [float (x) for x in dist_input.readlines ()]

    with open (multi_file) as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open (err_file) as err_input:
        err_multi = [float (x) for x in err_input.readlines ()]


    # Fit 1/r

    ls1 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        func1
                        )
    m1 = Minuit (ls1,
                 a = 5,
                 b = 2.58,
                 c = 0.,
                 d = 4,
                 k = 50
                 )

    m1.migrad ()
    m1.hesse ()

    a1 = m1.values["a"]
    b1 = m1.values["b"]
    c1 = m1.values["c"]
    d1 = m1.values["d"]
    k1 = m1.values["k"]

    print ("Modello: k / x + a * cos(d * x + c) + b")
    for par, val, err in zip (m1.parameters, m1.values, m1.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")


    chi2_1 = m1.fval
    ndof = m1.ndof
    chi2rid_1 = chi2_1 / ndof

    print ("chi2: ", chi2_1)
    print ("gradi di libertà: ", ndof)
    print ("chi2 ridotto: ", chi2rid_1)

    p = chi2.sf (chi2_1, ndof)
    print ("p value: ", p)

    # Fit 1/r^2

    ls1_q = LeastSquares (distanze,
                        multi,
                        err_multi,
                        func1_quadro
                        )
    m1_q = Minuit (ls1_q,
                 a = 5,
                 b = 2.58,
                 c = 0.,
                 d = 4,
                 k = 50
                 )

    m1_q.migrad ()
    m1_q.hesse ()

    a1_q = m1_q.values["a"]
    b1_q = m1_q.values["b"]
    c1_q = m1_q.values["c"]
    d1_q = m1_q.values["d"]
    k1_q = m1_q.values["k"]

    print ("Modello: k / x^2 + a * cos(d * x + c) + b")
    for par, val, err in zip (m1_q.parameters, m1_q.values, m1_q.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")


    chi2_1_q = m1_q.fval
    ndof = m1_q.ndof
    chi2rid_1_q = chi2_1_q / ndof

    print ("chi2: ", chi2_1_q)
    print ("gradi di libertà: ", ndof)
    print ("chi2 ridotto: ", chi2rid_1_q)

    p = chi2.sf (chi2_1_q, ndof)
    print ("p value: ", p)



    # Parabola

    ls2 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        func2
                        )
    m2 = Minuit (ls2,
                 a = 300,
                 b = 2.58,
                 c = 0.,
                 d = 4,
                 e = 0.
                 )

    m2.migrad ()
    m2.hesse ()

    a2 = m2.values["a"]
    b2 = m2.values["b"]
    c2 = m2.values["c"]
    d2 = m2.values["d"]
    e2 = m2.values["e"]

    print ("Modello: a * x^2 + b * x + c * cos(d * x) + e")
    for par, val, err in zip (m2.parameters, m2.values, m2.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")


    chi2_2 = m2.fval
    ndof = m2.ndof
    chi2rid_2 = chi2_2 / ndof

    print ("chi2: ", chi2_2)
    print ("gradi di libertà: ", ndof)
    print ("chi2 ridotto: ", chi2rid_2)

    p = chi2.sf (chi2_2, ndof)
    print ("p value: ", p)



    # Fit func3

    ls3 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        func3
                        )
    m3 = Minuit (ls3,
                 m = -0.02,      # decrescita lenta
                 q = 3,          # offset alto
                 A = 0.05,       # piccola oscillazione
                 d = 4,          # frequenza
                 c = 0
                 )

    m3.migrad ()
    m3.hesse ()

    m_1 = m3.values["m"]
    q_1 = m3.values["q"]
    A_1 = m3.values["A"]
    d_1 = m3.values["d"]
    c_1 = m3.values["c"]

    print ("Modello: m * x + q + A * cos(d * x + c)")
    for par, val, err in zip (m3.parameters, m3.values, m3.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    chi2_3 = m3.fval
    ndof = m3.ndof
    chi2rid_3 = chi2_3 / ndof

    print ("chi2: ", chi2_3)
    print ("gradi di libertà: ", ndof)
    print ("chi2 ridotto: ", chi2rid_3)

    p = chi2.sf (chi2_3, ndof)
    print ("p value: ", p)



    # Plot punti

    fig, ax = plt.subplots ()

    ax.set_title ("Andamento $S(r)$")
    ax.set_xlabel ("distanza $r$ (cm)")
    ax.set_ylabel ("differenza di potenziale $\\Delta V$ (V)")

    x_axis = np.linspace (min(distanze), max(distanze), 200)

    ax.errorbar (distanze,
                 multi,
                 yerr = err_multi,
                 linestyle = "None",
                 marker = "o",
                 color = "rebeccapurple",
                 label = "Dati osservati",
                 capsize = 4
                 )

    ax.plot (x_axis,
             [func1(x, a1, b1, c1, d1, k1) for x in x_axis],
             label = "$S = A/r + B cos(ωr + φ) + q$",
             color = "steelblue"
             )

    ax.plot (x_axis,
             [func1_quadro (x, a1_q, b1_q, c1_q, d1_q, k1_q) for x in x_axis],
             label = "$S = A/r^2 + B cos(ωr + φ) + q$",
             color = "deeppink"
             )

    ax.plot (x_axis,
             [func2 (x, a2, b2, c2, d2, e2) for x in x_axis],
             label = "$S = A r^2 + B r + C cos(ωr + φ) + q$",
             color = "firebrick"
             )

    ax.plot (x_axis,
             [func3 (x, m_1, q_1, A_1, d_1, c_1) for x in x_axis],
             label = "$S = - A r + B cos(ωr + φ) + q$",
             color = "purple"
             )

    plt.legend ()
    plt.show ()

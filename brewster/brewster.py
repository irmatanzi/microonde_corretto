import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from scipy.signal import find_peaks

def parabola (x, a, b, c):
    return a * x**2 + b * x + c

if __name__ == "__main__":

    with open ("angoli_incidenza.txt") as angoli_input:
        deg = [float(x) for x in angoli_input.readlines ()]

    with open ("multi.txt") as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open ("err_multi.txt") as err_input:
        err_multi = [float (x) for x in err_input.readlines ()]


    # Fit 

    ls = LeastSquares (deg,
                       multi,
                       err_multi,
                       parabola
                       )

    m = Minuit (ls,
                a = 1,
                b = 1, 
                c = 0
                )

    m.migrad ()
    m.hesse ()

    a1 = m.values["a"]
    b1 = m.values["b"]
    c1 = m.values["c"]

    for par, val, err in zip (m.parameters, m.values, m.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    p = chi2.sf (m.fval, m.ndof)
    print ("p value: ", p)


    # Massimo

    x_dense = np.linspace(min (deg), max (deg), 2000)

    N = 500

    primi_x = []

    primi_y = []

    for _ in range(N):
        # campionamento parametri
        a_sim = np.random.normal (a1, m.errors["a"])
        b_sim = np.random.normal (b1, m.errors["b"])
        c_sim = np.random.normal (c1, m.errors["c"])
        
        y_sim = [parabola (x, a_sim, b_sim, c_sim) for x in x_dense]
        peaks_sim, _ = find_peaks (y_sim)

        if len(peaks_sim) >= 1:
            x1 = x_dense[peaks_sim[0]]
            y1 = y_sim[peaks_sim[0]]

            primi_x.append(x1)
            primi_y.append(y1)

    print("\nRisultati con incertezza (dal toy)")

    print("\nPrimo massimo:")
    print(f"x = {np.mean(primi_x):.3f} ± {np.std(primi_x):.3f}")
    print(f"y = {np.mean(primi_y):.3f} ± {np.std(primi_y):.3f}")


    # Grafico

    fig, ax = plt.subplots ()

    ax.set_title ("Andamento di $S (\\alpha)$")
    ax.set_xlabel ("$\\alpha$ (rad)")
    ax.set_ylabel ("differenza di potenziale $\\Delta V$ (V)")

    x_axis = np.linspace (min (deg), max (deg), 200)

    ax.errorbar (deg,
                 multi,
                 yerr = err_multi,
                 marker = "o",
                 capsize = 4,
                 linestyle = "None",
                 label = "Dati osservati",
                 color = "navy"
                 )

    ax.plot (x_axis,
             [parabola (x, a1, b1, c1) for x in x_axis],
             label = "$S \\propto a x^2 + b x$",
             color = "firebrick"
             )

    plt.legend ()
    plt.show ()

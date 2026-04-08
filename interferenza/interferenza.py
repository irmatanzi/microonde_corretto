import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from scipy.signal import find_peaks

def func (x, a, b, c, d):
    return a * np.cos(d*x + c) + b

if __name__ == "__main__":

    with open ("angoli.txt") as angoli_input:
        angoli = [float (x) for x in angoli_input.readlines ()]

    with open ("multi.txt") as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open ("err_multi.txt") as err_input:
        err_multi = [float (x) for x in err_input.readlines ()]

    ls = LeastSquares (angoli,
                       multi,
                       err_multi,
                       func
                       )

    m = Minuit (ls,
                a = 0.3,
                b = 0.,
                c = 0.,
                d = 2 * np.pi / 0.349
                )

    m.fixed["c"] = True

    m.migrad ()
    m.hesse ()

    a = m.values["a"]
    b = m.values["b"]
    c = m.values["c"]
    d = m.values["d"]

    for par, val, err in zip (m.parameters, m.values, m.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    chi2rid = m.fval / m.ndof
    print (f"chi2: {m.fval}\nndof: {m.ndof}\nchi2 ridotto: {chi2rid}")
    p = chi2.sf (m.fval, m.ndof)
    print (f"p value: {p}")


    # Stima massimi con toy experiments

    N = 500

    x_dense = np.linspace (min (angoli), max (angoli), 2000)

    primi_x = []
    secondi_x = []

    primi_y = []
    secondi_y = []

    for _ in range (N):
        a_sim = np.random.normal (a, m.errors["a"])
        b_sim = np.random.normal (b, m.errors["b"])
        c_sim = np.random.normal (c, m.errors["c"])
        d_sim = np.random.normal (d, m.errors["d"])

        y_sim = [func (x, a_sim, b_sim, c_sim, d_sim) for x in x_dense]
        peaks_sim, _ = find_peaks(y_sim)

        if len (peaks_sim) >= 2:
            # primo massimo
            x1 = x_dense[peaks_sim[0]]
            y1 = y_sim[peaks_sim[0]]

            # secondo massimo
            x2 = x_dense[peaks_sim[1]]
            y2 = y_sim[peaks_sim[1]]

            primi_x.append(x1)
            primi_y.append(y1)

            secondi_x.append(x2)
            secondi_y.append(y2)

    print("\nRisultati con incertezza (dal toy)")

    print("\nPrimo massimo:")
    print(f"x = {np.mean(primi_x):.3f} ± {np.std(primi_x):.3f}")
    print(f"y = {np.mean(primi_y):.3f} ± {np.std(primi_y):.3f}")

    print("\nSecondo massimo:")
    print(f"x = {np.mean(secondi_x):.3f} ± {np.std(secondi_x):.3f}")
    print(f"y = {np.mean(secondi_y):.3f} ± {np.std(secondi_y):.3f}")

    

    # Plot del fit

    fig, ax = plt.subplots ()
    
    ax.set_title ("Andamento $S (\\alpha)$")
    ax.set_xlabel ("$\\alpha$ (rad)")
    ax.set_ylabel ("differenza di potenziale $\\Delta V$ (V)")

    x_axis = np.linspace (min(angoli), max(angoli), 200)

    ax.errorbar (angoli,
                 multi,
                 yerr = err_multi,
                 marker = 'o',
                 label = "Dati osservati",
                 capsize = 4,
                 linestyle = "None",
                 color = "indigo"
                 )

    ax.plot (x_axis,
             [func(x, a, b, c, d) for x in x_axis],
             label = "$S \\propto cos(\\alpha)$",
             color = "steelblue"
             )

    plt.legend ()
    plt.show ()

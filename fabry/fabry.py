import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from scipy.signal import find_peaks

def func (x, A, ω, φ, q):
    return A * np.cos (ω * x + φ) + q


if __name__ == "__main__":

    with open ("distanze_fabry.txt") as file_distanze:
        distanze = [float (x) for x in file_distanze.readlines ()]

    with open ("multimetro_fabry.txt") as file_multimetro:
        multi = [float (x) for x in file_multimetro.readlines ()]

    with open ("multimetro_sigma_fabry.txt") as multi_file:
        multi_sigma = [float (x) for x in multi_file.readlines ()]

    # Minuit func

    ls = LeastSquares (distanze,
                       multi,
                       multi_sigma,
                       func
                       )

    m = Minuit (ls,
                A = 0.6,
                ω = 5,
                φ = -2.8,
                q = 1.97
                )

    m.migrad ()
    m.hesse ()

    A1 = m.values["A"]
    ω1 = m.values["ω"]
    φ1 = m.values["φ"]
    q1 = m.values["q"]

    for par, val, err in zip (m.parameters, m.values, m.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    p = chi2.sf (m.fval, m.ndof)
    print (f"chi2: {m.fval}\nndof: {m.ndof}\nchi2 ridotto: {m.fval / m.ndof}")
    print (f"p value: {p}")


    # Massimi
    N = 500

    x_dense = np.linspace(min   (distanze), max (distanze), 2000)

    primi_x = []
    secondi_x = []

    primi_y = []
    secondi_y = []

    for _ in range(N):
        # campionamento parametri
        A_sim = np.random.normal (A1, m.errors["A"])
        ω_sim = np.random.normal (ω1, m.errors["ω"])
        φ_sim = np.random.normal (φ1, m.errors["φ"])
        q_sim = np.random.normal (q1, m.errors["q"])
        
        y_sim = [func (x, A_sim, ω_sim, φ_sim, q_sim) for x in x_dense]
        peaks_sim, _ = find_peaks (y_sim)

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



    # Plot punti

    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (distanze), max (distanze), 200)

    ax.set_title ("Andamento $S (\\alpha)$")
    ax.set_xlabel ("distanza (cm)")
    ax.set_ylabel ("differenza di potenziale $\\Delta V$ (V)")

    ax.errorbar (distanze,
                 multi,
                 yerr = multi_sigma,
                 marker = "o",
                 capsize = 4,
                 linestyle = "None",
                 color = "navy",
                 label = "Dati osservati"
                 )

    ax.plot (x_axis,
             [func (x, A1, ω1, φ1, q1) for x in x_axis],
             label = "$S \\propto cos(\\alpha)$",
             color = "deeppink"
             )

    plt.legend ()
    plt.show ()

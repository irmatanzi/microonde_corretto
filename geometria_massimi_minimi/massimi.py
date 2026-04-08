import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from scipy.signal import find_peaks


def func1 (x, a, b, c, d, k):
    return a * (np.cos (d * x + c)) + k / x + b

def func1_quadro (x, a, b, c, d, k):
    return a * (np.cos (d * x + c)) + k / x**2 + b

def func2 (x, a, b, c, d, e, f):
    return a * x**2 + b * x + f * np.cos(d * x + c) + e

def func3 (x, m, q, A, d, c):
    return m * x + q + A * np.cos(d * x + c)

def func_somma (x, a, b, c, omega, phi, q):
    return a / x + b / x**2 + c * np.cos (omega * x + phi) + q


if __name__ == "__main__":

    with open ("distanze_massimi.txt") as dist_input:
        distanze = [float (x) for x in dist_input.readlines ()]

    with open ("multimetro_massimi.txt") as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open ("errori_multi_massimi.txt") as err_input:
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
                 k = 300
                 )

    m1.migrad ()
    m1.hesse ()

    a1 = m1.values["a"]
    b1 = m1.values["b"]
    c1 = m1.values["c"]
    d1 = m1.values["d"]
    k1 = m1.values["k"]

    print ("Modello: k/x + a cos(d x + c) + b")
    for par, val, err in zip (m1.parameters, m1.values, m1.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    chi2_1 = m1.fval
    ndof = m1.ndof
    chi2rid_1 = chi2_1 / ndof

    print ("chi2: ", chi2_1)
    print ("ndof: ", ndof)
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
                 a = 300,
                 b = 2.58,
                 c = 0.,
                 d = 4,
                 k = 300
                 )

    m1_q.migrad ()
    m1_q.hesse ()

    a1_q = m1_q.values["a"]
    b1_q = m1_q.values["b"]
    c1_q = m1_q.values["c"]
    d1_q = m1_q.values["d"]
    k1_q = m1_q.values["k"]

    print ("Modello: k/x^2 + a cos(d x + c) + b")
    for par, val, err in zip (m1_q.parameters, m1_q.values, m1_q.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    chi2_1_q = m1_q.fval
    ndof = m1_q.ndof
    chi2rid_1_q = chi2_1_q / ndof

    print ("chi2: ", chi2_1_q)
    print ("ndof: ", ndof)
    print ("chi2 ridotto: ", chi2rid_1_q)

    p = chi2.sf (chi2_1_q, ndof)
    print ("p value: ", p)


    # Fit somma

    ls_somma = LeastSquares (distanze,
                             multi,
                             err_multi,
                             func_somma
                             )

    ms = Minuit (ls_somma,
                 a = 480,
                 b = 20000,
                 c = 0.2,
                 omega = 4.5,
                 phi = 0,
                 q = 0
                 )

    ms.migrad ()
    ms.hesse ()

    a_s = ms.values["a"]
    b_s = ms.values["b"]
    c_s = ms.values["c"]
    omega = ms.values["omega"]
    phi = ms.values["phi"]
    q_s = ms.values["q"]

    p = chi2.sf (ms.fval, ms.ndof)
    print ("Modello somma: ")
    print ("chi2: ", ms.fval)
    print ("ndof: ", ms.ndof)
    print ("p value: ", p)


    # Fit func2 (parabola)

    ls2 = LeastSquares (distanze,
                        multi,
                        err_multi,
                        func2
                        )
    m2 = Minuit (ls2,
                 a = 0.01,
                 b = 2.58,
                 c = 0.,
                 d = 4,
                 e = 0.,
                 f = 0
                 )

    m2.migrad ()
    m2.hesse ()

    a2 = m2.values["a"]
    b2 = m2.values["b"]
    c2 = m2.values["c"]
    d2 = m2.values["d"]
    e2 = m2.values["e"]
    f2 = m2.values["f"]

    print ("Modello: a x^2 + b x + f cos(d x + c) + e")
    for par, val, err in zip (m2.parameters, m2.values, m2.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    chi2_2 = m2.fval
    ndof = m2.ndof
    chi2rid_2 = chi2_2 / ndof

    print ("chi2: ", chi2_2)
    print ("ndof: ", ndof)
    print ("chi2 ridotto: ", chi2rid_2)

    p = chi2.sf (chi2_2, ndof)
    print ("p value: ", p)



    # Fit func3 (lineare)

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

    print ("Modello: m x + q + A cos(d x + c)")
    for par, val, err in zip (m3.parameters, m3.values, m3.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    chi2_3 = m3.fval
    ndof = m3.ndof
    chi2rid_3 = chi2_3 / ndof

    print ("chi2: ", chi2_3)
    print ("ndof: ", ndof)
    print ("chi2 ridotto: ", chi2rid_3)

    p = chi2.sf (chi2_3, ndof)
    print ("p value: ", p)

    # Ricerca dei massimi reali
    
    def f_fit (x): # usiamo la func1 tanto sono tutte uguali qui...
        return func1 (x, a1, b1, c1, d1, k1)

    # griglia fine
    x_dense = np.linspace(min   (distanze), max (distanze), 2000)
    y_dense = f_fit(x_dense)

    # trova massimi con scipy
    peaks, _ = find_peaks (y_dense)

    x_max = x_dense[peaks]
    y_max = y_dense[peaks]

    print("\nMassimi trovati:")
    for xm, ym in zip(x_max, y_max):
        print(f"x = {xm:.3f}, y = {ym:.3f}")

    # simulazione monte carlo per i massimi
    N = 500

    primi_x = []
    secondi_x = []

    primi_y = []
    secondi_y = []

    for _ in range(N):
        # campionamento parametri
        a_sim = np.random.normal (a1, m1.errors["a"])
        b_sim = np.random.normal (b1, m1.errors["b"])
        c_sim = np.random.normal (c1, m1.errors["c"])
        d_sim = np.random.normal (d1, m1.errors["d"])
        k_sim = np.random.normal (k1, m1.errors["k"])

        y_sim = [func1 (x, a_sim, b_sim, c_sim, d_sim, k_sim) for x in x_dense]
        peaks_sim, _ = find_peaks(y_sim)

        if len(peaks_sim) >= 2:
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
             [func_somma (x, a_s, b_s, c_s, omega, phi, q_s) for x in x_axis],
             label = "$S = A/r + B/r^2 + C cos(ωr + φ) + q$",
             color = "royalblue"
             )

    ax.plot (x_axis,
             [func2 (x, a2, b2, c2, d2, e2, f2) for x in x_axis],
             label = "$S = A r^2 + B r + C cos(ωr + φ) + q$",
             color = "firebrick"
             )

    ax.plot (x_axis,
             [func3 (x, m_1, q_1, A_1, d_1, c_1) for x in x_axis],
             label = "$S = - A r + B cos(ωr + φ) + q$",
             color = "purple"
             )

    plt.legend (loc = "lower left")
    plt.show ()

    


    

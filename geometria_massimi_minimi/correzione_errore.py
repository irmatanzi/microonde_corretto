# Dal confronto tra i grafici dei residui dei vari fit, sembra che ci sia un errore sistematico che non è stato considerato. 
# In particolare, i residui sembrano essere più grandi di quanto ci si aspetterebbe in media, e questo potrebbe essere dovuto 
# a un errore di calibrazione o a un altro fattore che non è stato preso in considerazione.  
# In generale sembra che i residui siano distribuiti in modo giustamente casuale per 1/r^2, 
# quindi consideriamo quello come il modello più adatto.

# Ricerca di fattore moltiplicativo per correggere l'errore sistematico:
# L'idea è di scalare gli errori in modo che il chi2 ridotto sia circa 1, 
# e vedere come cambiano i risultati dei fit e dei massimi


import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from pathlib import Path

# Stesse funzioni di prima senza func2 (parabola) che fa veramente schifo.

def func1 (x, a, b, c, d, k):
    return k / x + a * np.cos(d * x + c) + b

def func1_quadro (x, a, b, c, d, k):
    return k / x**2 + a * np.cos(d * x + c) + b

def func_somma (x, a, b, c, d, k_1, k_2):
    return k_1 / x + k_2 / x**2 + a * np.cos(d * x + c) + b

def func3 (x, m, q, A, d, c):
    return m * x + q + A * np.cos(d * x + c)

def correggi_errore (err, chi2rid):
    fattore = np.sqrt (chi2rid)
    return [e * fattore for e in err]


if __name__ == "__main__":

    current_dir = Path (__file__).parent
    dist_file = current_dir / "dist.txt"
    multi_file = current_dir / "multimetro.txt"
    err_file = current_dir / "multi_sigmaunif.txt"

    with open (dist_file) as dist_input:
        distanze = [float (x) for x in dist_input.readlines ()]

    with open (multi_file) as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open (err_file) as err_input:
        err_multi = [float (x) for x in err_input.readlines ()]

    # Per il conto usiamo il k di 1/r^2 perché sembra il fit migliore

    chi2rid_1 = 36.64  # Sostituire con il valore effettivo del chi2 ridotto ottenuto dal fit
    err_multi_corretto = correggi_errore (err_multi, chi2rid_1)
    print (f"Fattore di correzione: {np.sqrt(chi2rid_1):.2f}") # ris: 1.87
    
    # Ora rifacciamo i fit utilizzando err_multi_corretto al posto di err_multi

    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    
    # Fit con 1/r

    ls1 = LeastSquares (distanze,
                        multi,
                        err_multi_corretto,
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

    # Fit con 1/r^2

    ls1_q = LeastSquares (distanze,
                          multi,
                          err_multi_corretto,
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

    #Fit 1/r + 1/r^2

    ls_somma = LeastSquares (distanze,
                             multi,
                             err_multi_corretto,
                             func_somma
                             )

    m_somma = Minuit (ls_somma,
                 a = 5,
                 b = 2.58,
                 c = 0.,
                 d = 4,
                 k_1 = 50,
                 k_2 = 50
                 )

    m_somma.migrad ()
    m_somma.hesse ()

    a_somma = m_somma.values["a"]
    b_somma = m_somma.values["b"]
    c_somma = m_somma.values["c"]
    d_somma = m_somma.values["d"]
    k1_somma = m_somma.values["k_1"]
    k2_somma = m_somma.values["k_2"]

    print ("Modello: k_1 / x + k_2 / x^2 + a * cos(d * x + c) + b")
    for par, val, err in zip (m_somma.parameters, m_somma.values, m_somma.errors):
        print (f"{par}: {val:.3f} ± {err:.3f}")

    chi2_somma = m_somma.fval
    ndof = m_somma.ndof
    chi2rid_somma = chi2_1 / ndof

    print ("chi2: ", chi2_somma)
    print ("gradi di libertà: ", ndof)
    print ("chi2 ridotto: ", chi2rid_somma)

    p = chi2.sf (chi2_somma, ndof)
    print ("p value: ", p)


    # Fit func3

    ls3 = LeastSquares (distanze,
                        multi,
                        err_multi_corretto,
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
                 yerr = err_multi_corretto,
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
             [func_somma (x, a_somma, b_somma, c_somma, d_somma, k1_somma, k2_somma) for x in x_axis],
             label = "$S = A/r + B/r^2 + C cos(ωr + φ) + q$",
             color = "royalblue"
             )

    ax.plot (x_axis,
             [func3 (x, m_1, q_1, A_1, d_1, c_1) for x in x_axis],
             label = "$S = - A r + B cos(ωr + φ) + q$",
             color = "purple"
             )

    plt.legend ()
    plt.show ()
    









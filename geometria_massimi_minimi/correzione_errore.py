# Dal confronto tra i grafici dei residui dei vari fit, sembra che ci sia un errore sistematico che non è stato considerato. In particolare, i residui sembrano essere più grandi di quanto ci si aspetterebbe in media, e questo potrebbe essere dovuto a un errore di calibrazione o a un altro fattore che non è stato preso in considerazione.  
# In generale sembra che i residui siano distribuiti in modo giustamente casuale per 1/r^2, quindi consideriamo quello come il modello più adatto.

# Ricerca di fattore moltiplicativo per correggere l'errore sistematico:
# L'idea è di scalare gli errori in modo che il chi2 ridotto sia circa 1, e vedere come cambiano i risultati dei fit e dei massimi


import numpy as np
import matplotlib.pyplot as plt
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
    dist_file = current_dir / "distanze_massimi.txt"
    multi_file = current_dir / "multimetro_massimi.txt"
    err_file = current_dir / "multi_sigmaunif.txt"

    with open (dist_file) as dist_input:
        distanze = [float (x) for x in dist_input.readlines ()]

    with open (multi_file) as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open (err_file) as err_input:
        err_multi = [float (x) for x in err_input.readlines ()]

    chi2rid_1 = 3.5  # Sostituire con il valore effettivo del chi2 ridotto ottenuto dal fit
    err_multi_corretto = correggi_errore (err_multi, chi2rid_1)
    print (f"Fattore di correzione: {np.sqrt(chi2rid_1):.2f}")
    # Ora puoi rifare i fit utilizzando err_multi_corretto al posto di err_multi

    fig, ax = plt.subplots (nrows = 1, ncols = 1)

    









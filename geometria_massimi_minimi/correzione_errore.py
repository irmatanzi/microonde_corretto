# Dal confronto tra i grafici dei residui dei vari fit, sembra che ci sia un errore sistematico che non è stato considerato. In particolare, i residui sembrano essere più grandi di quanto ci si aspetterebbe in media, e questo potrebbe essere dovuto a un errore di calibrazione o a un altro fattore che non è stato preso in considerazione.  
# In generale sembra che i residui siano distribuiti in modo giustamente casuale per 1/r^2, quindi consideriamo quello come il modello più adatto.

# Ricerca di fattore moltiplicativo per correggere l'errore sistematico:
# L'idea è di scalare gli errori in modo che il chi2 ridotto sia circa 1, e vedere come cambiano i risultati dei fit e dei massimi

def correggi_errore (err, chi2rid):
    fattore = np.sqrt (chi2rid)
    return [e * fattore for e in err]


if __name__ == "__main__":

    with open ("distanze_massimi.txt") as dist_input:
        distanze = [float (x) for x in dist_input.readlines ()]

    with open ("multimetro_massimi.txt") as multi_input:
        multi = [float (x) for x in multi_input.readlines ()]

    with open ("distanze_sigmaunif.txt") as err_input:
        err_multi = [float (x) for x in err_input.readlines ()]

    chi2rid_1 = 3.5  # Sostituisci con il valore effettivo del chi2 ridotto ottenuto dal fit
    err_multi_corretto = correggi_errore (err_multi, chi2rid_1)

    # Ora puoi rifare i fit utilizzando err_multi_corretto al posto di err_multi



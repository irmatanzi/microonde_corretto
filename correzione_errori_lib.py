import numpy as np

# la libreria contiene le funzioni usate per la correzione degli errori sistematici nei fit

# In particolare la funzione correggi_errore che prende in input gli errori originali e il chi2 ridotto del fit, 
# e restituisce gli errori corretti moltiplicati per la radice del chi2 ridotto.

def correggi_errore (err:list, chi2rid:float) -> list:
    fattore = np.sqrt (chi2rid)
    return [e * fattore for e in err]
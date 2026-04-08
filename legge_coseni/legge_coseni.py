import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from math import sqrt
from scipy.stats import chi2


def I_cos (x, a, b, c):

    return a * np.cos(x + c) + b

def I_cos2 (x, a, b, c):

    return a * (np.cos(x + c)**2) + b

def I_cos3 (x, a, b, c):

    return a * (np.cos(x + c)**3) + b

def I_somme1 (x, a, b, c, d, e):

    return a * (np.cos (x + c)) + b * (np.cos(x + d)**2) + e

def I_somme2 (x, a, b, c, d, e):

    return a * (np.cos(x + c)**2) + b * (np.cos(x + c)**3) + e


if __name__ == "__main__":

    with open ("angoli_coseni.txt") as file_angoli:
        angoli = [float (x) for x in file_angoli.readlines ()]

    with open ("intensità_coseni.txt") as file_int:
        intensità = [float (x) * 30 for x in file_int.readlines ()]

    I_0 = intensità[0]
    sigma_intensità = np.ones(len(intensità)) * 0.02 * 30. / sqrt(12)

    coseni = [np.cos (x) for x in angoli]
    coseni_quadri = [x**2 for x in coseni]


    # Coseni fit

    ls_1 = LeastSquares (angoli,
                         intensità,
                         sigma_intensità,
                         I_cos
                         )

    myminuit_1 = Minuit (ls_1,
                         a = I_0,
                         b = 0.,
                         c = 0.
                         )

    myminuit_1.migrad ()
    myminuit_1.hesse ()

    print ("Fit con coseno:")

    a_1 = myminuit_1.values["a"]
    b_1 = myminuit_1.values["b"]
    c_1 = myminuit_1.values["c"]

    for par in myminuit_1.parameters:
        val = myminuit_1.values[par]
        err = myminuit_1.errors[par]
        print(f"{par} = {val:.3f} ± {err:.3f}")

    chi2_1 = myminuit_1.fval
    ndof_1 = len(angoli) - myminuit_1.nfit
    chi2rid_1 = chi2_1 / ndof_1

    print("Chi2 = ", chi2_1)
    print("ndof = ", ndof_1)
    print("Chi2 ridotto = ", chi2rid_1)

    p = chi2.sf (chi2_1, ndof_1)
    print ("p value: ", p)


    # Coseni^2 fit

    ls_2 = LeastSquares (angoli,
                         intensità,
                         sigma_intensità,
                         I_cos2
                         )

    myminuit_2 = Minuit (ls_2,
                         a = I_0,
                         b = 0.,
                         c = 0.
                         )

    myminuit_2.migrad ()
    myminuit_2.hesse ()

    print ("Fit con coseno quadro: ")

    a_2 = myminuit_2.values["a"]
    b_2 = myminuit_2.values["b"]
    c_2 = myminuit_2.values["c"]

    for par, val, err in zip (myminuit_2.parameters, 
                              myminuit_2.values, 
                              myminuit_2.errors):
        print (f"{par} = {val:.3f} ± {err:.3f}")

    chi2_2 = myminuit_2.fval
    ndof_2 = len(angoli) - myminuit_2.nfit
    chi2rid_2 = chi2_2 / ndof_2

    print("Chi2 =", chi2_2)
    print("ndof =", ndof_2)
    print("Chi2 ridotto =", chi2rid_2)

    p = chi2.sf (chi2_2, ndof_2)
    print ("p value: ", p)


    # Coseni^3 fit

    ls_3 = LeastSquares (angoli,
                         intensità,
                         sigma_intensità,
                         I_cos3
                         )

    myminuit_3 = Minuit (ls_3,
                         a = I_0,
                         b = 0.,
                         c = 0.
                         )

    myminuit_3.migrad ()
    myminuit_3.hesse ()

    print ("Fit con coseno cubo:")

    a_3 = myminuit_3.values["a"]
    b_3 = myminuit_3.values["b"]
    c_3 = myminuit_3.values["c"]

    for par in myminuit_3.parameters:
        val = myminuit_3.values[par]
        err = myminuit_3.errors[par]
        print(f"{par} = {val:.3f} ± {err:.3f}")

    chi2_3 = myminuit_3.fval
    ndof_3 = len(angoli) - myminuit_3.nfit
    chi2rid_3 = chi2_3 / ndof_3

    print("Chi2 = ", chi2_3)
    print("ndof = ", ndof_3)
    print("Chi2 ridotto = ", chi2rid_3)

    p = chi2.sf (chi2_3, ndof_3)
    print ("p value: ", p)


    # Coseni somme cos e cos^2

    ls_s1 = LeastSquares (angoli,
                         intensità,
                         sigma_intensità,
                         I_somme1
                         )

    myminuit_s1 = Minuit (ls_s1,
                         a = I_0,
                         b = I_0,
                         c = 0.,
                         d = 0.,
                         e = 0.
                         )

    myminuit_s1.migrad ()
    myminuit_s1.hesse ()

    print ("Fit con somma di coseno e coseno quadro:")

    a_s1 = myminuit_s1.values["a"]
    b_s1 = myminuit_s1.values["b"]
    c_s1 = myminuit_s1.values["c"]
    d_s1 = myminuit_s1.values["d"]
    e_s1 = myminuit_s1.values["e"]

    for par in myminuit_s1.parameters:
        val = myminuit_s1.values[par]
        err = myminuit_s1.errors[par]
        print(f"{par} = {val:.3f} ± {err:.3f}")

    chi2_s1 = myminuit_s1.fval
    ndof_s1 = len(angoli) - myminuit_s1.nfit
    chi2rid_s1 = chi2_s1 / ndof_s1

    print("Chi2 = ", chi2_s1)
    print("ndof = ", ndof_s1)
    print("Chi2 ridotto = ", chi2rid_s1)

    p = chi2.sf (chi2_s1, ndof_s1)
    print ("p value: ", p)


    # Fit somme cos^2 e cos^3

    ls_s2 = LeastSquares (angoli,
                         intensità,
                         sigma_intensità,
                         I_somme2
                         )

    myminuit_s2 = Minuit (ls_s2,
                         a = I_0,
                         b = I_0,
                         c = 0.,
                         d = 0.,
                         e = 0.
                         )

    myminuit_s2.migrad ()
    myminuit_s2.hesse ()

    print ("Fit con somma di coseno quadro e coseno cubo:")

    a_s2 = myminuit_s2.values["a"]
    b_s2 = myminuit_s2.values["b"]
    c_s2 = myminuit_s2.values["c"]
    d_s2= myminuit_s2.values["d"]
    e_s2 = myminuit_s2.values["e"]

    for par in myminuit_s2.parameters:
        val = myminuit_s2.values[par]
        err = myminuit_s2.errors[par]
        print(f"{par} = {val:.5f} ± {err}")

    chi2_s2 = myminuit_s2.fval
    ndof_s2 = len(angoli) - myminuit_s2.nfit
    chi2rid_s2 = chi2_s2 / ndof_s2

    print("Chi2 = ", chi2_s2)
    print("ndof = ", ndof_s2)
    print("Chi2 ridotto = ", chi2rid_s2)

    p = chi2.sf (chi2_s2, ndof_s2)
    print ("p value: ", p)




    # Grafico

    fig, ax = plt.subplots ()

    ax.set_title ("Andamento $I (\\alpha)$")
    ax.set_xlabel ("$\\alpha$ (rad)")
    ax.set_ylabel ("$I$ (mA)")

    x_axis = np.linspace (min (angoli), max (angoli), 200)

    ax.errorbar (angoli,
                 intensità,
                 color = "indigo",
                 label = "Dati misurati",
                 yerr = sigma_intensità,
                 linestyle = "None",
                 marker = "o",
                 capsize=4
                 )

    ax.plot (x_axis,
             [a_1 * np.cos(x + c_1) + b_1 for x in x_axis],
             color = "mediumpurple",
             label = "$I \\propto cos(\\alpha)$"
             )

    ax.plot (x_axis,
             [a_2 * np.cos(x + c_2)**2 + b_2 for x in x_axis],
             color = "crimson",
             label = "$I \\propto cos^2(\\alpha)$"
             )

    ax.plot (x_axis,
             [a_3 * np.cos(x + c_3)**3 + b_3 for x in x_axis],
             color = "mediumvioletred",
             label = "$I \\propto cos^3(\\alpha)$"
             )

    ax.plot (x_axis,
             [a_s1 * np.cos(x + c_s1) + b_s1 * np.cos(x + d_s1)**2 + e_s1 for x in x_axis],
             color = "cornflowerblue",
             label = "$I \\propto A cos(\\alpha) + B cos^2(\\alpha)$"
             )

    plt.legend ()
    plt.show ()

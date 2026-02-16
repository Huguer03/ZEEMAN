import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import h,c
import matplotlib.pyplot as plt
import utiles as ut
import scienceplots
plt.style.use(['science'])

def f(x,m,n):
    return m * x + n

def coef_correlacion_lineal(nu,B,popt):
    scr = np.sum((nu - f(B, *popt))**2) # suma de los cuadrados residual
    sct = np.sum((nu - np.mean(nu))**2)              # suma cuadrados total
    return 1 - scr/sct

if __name__ == "__main__":
    I_err = 0.01
    t     = 3e-3

    regresion_df = pd.read_excel("calibrado_terminado.ods", engine="odf")
    data_df = pd.read_excel("mediciones.ods", engine="odf")

    mb, nb, mb_err, nb_err = regresion_df.at[0,"m"], regresion_df.at[0,"n"], regresion_df.at[0,"dm"], regresion_df.at[0,"dn"]
    I   = data_df["I(A)"].to_numpy()
    rm1 = data_df["r-1(um)"].to_numpy()
    rp1 = data_df["r+1(um)"].to_numpy()
    rm2 = data_df["r-2(um)"].to_numpy()
    rp2 = data_df["r+2(um)"].to_numpy()

    B = f(I,mb,nb)
    B_err = np.sqrt((I*mb_err)**2 + (mb*I_err)**2 + nb_err**2)

    resultados_df = pd.DataFrame({"B(T)": B, "dB(T)": B_err})
    resultados_df["B(T)"], resultados_df["dB(T)"] = ut.redondear_vectorizado(resultados_df["B(T)"], resultados_df["dB(T)"])

    d  = 0.5 * (rp1**2 - rm1**2 + rp2**2 - rm2**2)
    D  = 0.5 * (rp2**2 - rp1**2 + rm2**2 - rm1**2)
    nu = d / (2 * t * D)

    resultados_df["nu(m-1)"] = nu

    popt, pcov = curve_fit(f,B, nu, sigma=B_err)

    m, n         = popt
    m_err, n_err = np.sqrt(np.diag(pcov))

    magneton_bohr     = 0.5 * h * c * m
    magneton_bohr_err = 0.5 * h * c * m_err

    resultados_df.at[0,"m"], resultados_df.at[0,"dm"] = ut.redondear_escalares(m, m_err)
    resultados_df.at[0,"n"], resultados_df.at[0,"dn"] = ut.redondear_escalares(n, n_err)
    resultados_df.at[0,"R2"] = coef_correlacion_lineal(nu,B,popt)
    resultados_df.at[0,"magneton_bohr"], resultados_df.at[0,"magneton_bohr_err"] = ut.redondear_escalares(magneton_bohr, magneton_bohr_err)

    resultados_df.to_excel("resultados.ods", engine="odf")


    # Plot
    plt.scatter(B, nu, label='Datos experimentales', color='red')
    plt.plot(B, f(B, *popt), label=r'Ajuste $\Delta \nu$')
    plt.xlabel(r'$B(T)$')
    plt.ylabel(r'$\Delta \nu (m^{-1})$')
    plt.legend()
    plt.grid(True)
    plt.show()

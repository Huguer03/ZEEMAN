import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import utiles as ut
import scienceplots
plt.style.use(['science'])

def f(x,m,n):
    return m * x + n

def coef_correlacion_lineal(B,I,popt):
    scr = np.sum((B - f(I, *popt))**2) # suma de los cuadrados residual
    sct = np.sum((B - np.mean(B))**2)              # suma cuadrados total
    return 1 - scr/sct

if __name__ == "__main__":
    # Extraemos los datos
    datos_df = pd.read_excel("B(I).ods", engine="odf")

    I = datos_df["I(A)"].to_numpy()
    B = datos_df["B(T)"].to_numpy() * 1e-3

    # Regresion lineal
    popt, pcov = curve_fit(f,I,B)

    # Los parametros optimos
    m_opt, n_opt = popt
    m_err, n_err = np.sqrt(np.diag(pcov))

    # Pasamos los datos
    datos_df.at[0,"m"], datos_df.at[0,"dm"] = ut.redondear_escalares(m_opt,m_err)
    datos_df.at[0,"n"], datos_df.at[0,"dn"] = ut.redondear_escalares(n_opt,m_err)
    datos_df.at[0,"R2"] = coef_correlacion_lineal(B,I,popt)

    datos_df.to_excel("calibrado_terminado.ods", engine="odf")

    # Plot
    plt.scatter(I, B, label='Datos experimentales', color='red')
    plt.plot(I, f(I, *popt), label=r'Ajuste $B(I)$')
    plt.xlabel(r'$I (A)$')
    plt.ylabel(r'$B (T)$')
    plt.legend()
    plt.grid(True)
    plt.show()




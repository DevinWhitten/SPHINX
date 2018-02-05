import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

import sys, os
sys.path.append("/Users/MasterD/Google Drive/JPLUS/Pipeline2.0/Temperature/interface")
import temperature_functions
def GAUSS(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))


matplotlib.rcParams['axes.linewidth'] = 1.2

plt.ion()
################################################################################
##### NEW Network Test for TEMP
TEMP = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Pipeline2.0/Temperature/output/net_temp_mag_test.csv")
print("# of Temp stars:  ", len(TEMP))
TEMP_Result = TEMP['NET_TEFF'] #pd.read_csv("Networks/T_4000_7000/TEFF_Test_Res.dat", header=None)
TEMP_True = TEMP['TEFF'] #pd.read_csv("Networks/T_4000_7000/TEFF_Test_Acc.dat", header=None)

print("Residual Average:  ", np.average(TEMP_Result - TEMP_True))
TEMP_Result = TEMP_Result - np.average(TEMP_Result - TEMP_True)
TEMP = pd.DataFrame({"Result":TEMP_Result, "Accepted": TEMP_True, "Residual":TEMP_Result - TEMP_True})



FEH = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Preliminary/SEGUE Pipeline/Metallicity/output/net_sklearn_feh_mag_valid.csv")
#Native_FEH = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Preliminary/SEGUE Pipeline/Metallicity/output/net_sklearn_feh_mag_native.csv")
#Synth_FEH = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Preliminary/SEGUE Pipeline/Metallicity/output/net_sklearn_feh_mag_test.csv")

Native_FEH = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Pipeline2.0/Metallicity/output/net_sklearn_feh_mag_native.csv")
Synth_FEH = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Pipeline2.0/Metallicity/output/net_sklearn_feh_mag_synthetic.csv")


FEH = pd.DataFrame({"Result":FEH['NET_FEH'], "Accepted":FEH['FEH'], "Residual":FEH['NET_FEH'] - FEH['FEH']})

Native_FEH = pd.DataFrame({"Result":Native_FEH['NET_FEH'], "Accepted":Native_FEH['FEH'],
                           "Residual":Native_FEH['NET_FEH'] - Native_FEH['FEH']})

Synth_FEH = pd.DataFrame({"Result":Synth_FEH['NET_FEH'], "Accepted":Synth_FEH['FEH'],
                           "Residual":Synth_FEH['NET_FEH'] - Synth_FEH['FEH']})

FEH_MIN = min(Synth_FEH['Accepted'])
FEH_MAX = max(Synth_FEH['Accepted'])
########################### OLD ############################
#FEH_Result =  pd.read_csv("Networks/FEH/T_4000_7000_FEH_Test_Res_RPROP.dat", header= None)
#FEH_True = pd.read_csv("Networks/FEH/T_4000_7000_FEH_Test_Acc_RPROP.dat", header=None)
#FEH = pd.DataFrame({"Result":FEH_Result[0], "Accepted":FEH_True[0], "Residual":FEH_Result[0] - FEH_True[0]})
##############################################################

print("# of FEH stars:   ", len(FEH))
print('# of Native FEH stars:   ', len(Native_FEH))
pp = PdfPages("plots/TEFF_FEH.pdf")
fig = plt.figure(figsize=(10,3))

BOTTOM = 0.12
TOP = 0.5
LEFT = 0.1
RIGHT = 0.54
Y_SPAN = 0.38
X_SPAN = 0.38

LEFT_Top = plt.axes([LEFT, TOP, X_SPAN, Y_SPAN])
LEFT_Bottom = plt.axes([LEFT, BOTTOM, X_SPAN, Y_SPAN], sharex=LEFT_Top)
LEFT_Hist = plt.axes([0.05, BOTTOM, 0.05, Y_SPAN], sharey=LEFT_Bottom)

RIGHT_Top = plt.axes([RIGHT, TOP, X_SPAN, Y_SPAN])
RIGHT_Bottom = plt.axes([RIGHT, BOTTOM, X_SPAN, Y_SPAN], sharex=RIGHT_Top)
RIGHT_Hist = plt.axes([0.92, BOTTOM, 0.05, Y_SPAN], sharey=RIGHT_Bottom)

Handles = [LEFT_Top, LEFT_Bottom, LEFT_Hist, RIGHT_Top, RIGHT_Bottom, RIGHT_Hist]


SIZE = 0.5

LEFT_Bottom.plot(np.linspace(4000,8000, 50), np.zeros(50), linestyle="--", zorder=1)
LEFT_Top.plot(np.linspace(4000,8000,50), np.linspace(4000,8000,50), linestyle="--", zorder=1)

RIGHT_Bottom.plot(np.linspace(-3.5, -0.5,50), np.zeros(50), linestyle="--", zorder=1)
RIGHT_Top.plot(np.linspace(-3.5, -0.5,50), np.linspace(-3.5, -0.5,50), linestyle="--", zorder=1)


LEFT_Top.scatter(TEMP['Accepted'], TEMP['Result'], s=SIZE, color="Black", alpha=0.6, zorder=3)
TEMP_RES = LEFT_Bottom.scatter(TEMP['Accepted'], TEMP['Residual'], s=SIZE, color="Black", alpha=0.6, zorder=3)

#RIGHT_Top.scatter(FEH['Accepted'], FEH['Result'], s=SIZE, color="Black", alpha=0.6, zorder=3)
#FEH_RES = RIGHT_Bottom.scatter(FEH['Accepted'], FEH['Residual'], s=SIZE, color="Black", alpha=0.6, zorder=3)

RIGHT_Top.scatter(Native_FEH['Accepted'], Native_FEH['Result'], s=3, color="red", alpha=0.6, zorder=3)
Native_RES = RIGHT_Bottom.scatter(Native_FEH['Accepted'], Native_FEH['Residual'], s=3, color="red", alpha=0.6, zorder=3)

RIGHT_Top.scatter(Synth_FEH['Accepted'], Synth_FEH['Result'], s=15,
                    marker="+", color="black", alpha=0.75, zorder=3)

Synth_RES = RIGHT_Bottom.scatter(Synth_FEH['Accepted'], Synth_FEH['Residual'], s=15,
                    marker="+", color="black", alpha=0.75, zorder=15)


LEFT_Top.set_ylim([3800,8000])
LEFT_Top.set_xlim([4000,8000])

LEFT_Bottom.set_ylim([-750, 750])


RIGHT_Top.set_xlim([-2.50, -1.0])
RIGHT_Bottom.set_xlim([-2.50, -1.0])

RIGHT_Top.set_ylim([-2.50, -1.0])

RIGHT_Bottom.set_ylim([-0.6,0.6])


LEFT_Bottom.set_xlabel("Teff$_{Acc}$ [K]", family="Serif", fontsize=11, labelpad=0.5)
LEFT_Top.set_ylabel("Teff$_{Net}$ [K]", family="Serif", fontsize=11)

RIGHT_Bottom.set_xlabel("[Fe/H]$_{Acc}$", family="Serif", fontsize=11, labelpad=0.5)
RIGHT_Top.set_ylabel("[Fe/H]$_{Net}$", family="Serif", fontsize=11, labelpad=0.5)
RIGHT_Bottom.set_ylabel("Residual", family="Serif", fontsize=12, labelpad=-5.0)

[label.tick_params(direction="in") for label in Handles]

############################ HISTOGRAM #########################################
TEMP_HIST = LEFT_Hist.hist(TEMP['Residual'], bins=15, orientation="horizontal",
                            normed=True, edgecolor="black", color="white")
FEH_HIST = RIGHT_Hist.hist(Native_FEH['Residual'], bins=25, orientation="horizontal",
                            normed=True, edgecolor="black", color="white")
LEFT_Hist.set_xlim([0.0035, 0.0])
#FEH_HIST = RIGHT_Hist.hist(FEH['Residual'], orientation="horizontal", bins=35)
LEFT_Hist.xaxis.set_ticks([])
RIGHT_Hist.xaxis.set_ticks([])
RIGHT_Bottom.set_yticks([-0.5, 0.0, 0.5])
RIGHT_Bottom.set_xticks([-2.5, -2.0, -1.5])


teff_popt, teff_pcov = temperature_functions.gaussian_sigma(TEMP['Residual'], bins=15)

print("TEMP SCALE:  ", teff_popt[2])
x_fit = np.linspace(min(TEMP['Residual']), max(TEMP['Residual']), 100)
y_fit = GAUSS(x_fit, teff_popt[0], teff_popt[1], teff_popt[2])
LEFT_Hist.plot(y_fit, x_fit, lw=1.25, color="r",alpha=0.75)

feh_popt, feh_pcov = temperature_functions.gaussian_sigma(FEH['Residual'], bins=15)
print("FEH SCALE:  ",feh_popt[2])


x_fit = np.linspace(min(FEH['Residual']), max(FEH['Residual']), 100)
y_fit = GAUSS(x_fit, feh_popt[0], feh_popt[1], feh_popt[2])
RIGHT_Hist.plot(y_fit, x_fit, lw=1.25, color="r",alpha=0.75)

#fig.legend((TEMP_RES, FEH_RES), ("$\Delta$Teff", "$\Delta$[Fe/H]"), loc=(0.5, 0.25), framealpha=0.99, borderaxespad=10000.)
LEFT_Bottom.text(4100, 450,"$\sigma$Teff=" + '%.0f' % (teff_popt[2]) +  "K", fontsize=11, family="Serif", bbox={'facecolor':'white', 'alpha':0.9, 'pad':4.0})
RIGHT_Bottom.text(-2.20, 0.35, "$\sigma$[Fe/H]= 0.18", fontsize=11, family="Serif", bbox={'facecolor':'white', 'alpha':0.9, 'pad':4.0})

plt.setp(RIGHT_Hist.get_yticklabels(),visible=False)
plt.setp(RIGHT_Top.get_xticklabels(), visible=False)
plt.setp(LEFT_Top.get_xticklabels(), visible=False)


plt.show()
input()
plt.show()
plt.savefig(pp, format='pdf')
pp.close()

plt.close()

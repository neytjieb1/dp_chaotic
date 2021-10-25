############################################
# Importing
#############################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns; sns.set()
import Libs.functions_plotting as fp
import Libs.GLOBAL as G

#############################################
# Plotting - setup
#############################################

def plots(dp):
    u_predicted=np.loadtxt('Data/predicted_{d}_{c}.txt'.format(d=dp, c=G.CTR))
    u_data=np.loadtxt('Data/actual_{d}_{c}.txt'.format(d=dp, c=G.CTR))


    # u_predicted = np.reshape(u_predicted, (u_predicted.shape[0], 1))    #if 1D
    # u_data = np.reshape(u_data, (u_data.shape[0], 1))


    # equal lengths
    if u_predicted.shape[0] - u_data.shape[0] >0 :
        u_predicted = u_predicted[:u_data.shape[0], :]
    elif u_predicted.shape[0] - u_data.shape[0] < 0 :
        u_data = u_data[:u_predicted.shape[0], :]


    ### SETUP 
    plt.rcParams['figure.figsize'] = [15, 4]

    spec = gridspec.GridSpec( nrows=4,ncols=1,
                            hspace=0.2,height_ratios=[0.75,0.75, 0.75, 0.75])

    _rows = 4
    _cols = 1
    fig=plt.figure()
    axes = []
    for i in range(_cols):
        for j in range(_rows):
            axes.append(fig.add_subplot(spec[j,i]))

    savingdata = input("Plots to be generated. Save plots?\n")=='y'

    # savingdata = False

    #Plotting FIG F
    # fp.figF(axes, fig, spec, u_data, u_predicted, savingdata, rows= 2, cols=2, p3=200)
    fp.figF(axes, fig, u_data, u_predicted, savingdata, rows= _rows, cols=_cols)

    # Plotting FIG C
    # fig=plt.figure()
    # spec = gridspec.GridSpec( nrows=2,ncols=1,
    #                          hspace=0.2,height_ratios=[0.75,0.75])
    # axs0=fig.add_subplot(spec[0])
    # axs1=fig.add_subplot(spec[1])
    # fp.figC(axs0, axs1, fig, spec, u_data, u_predicted, savingdata)    

    #Plotting FIG A 
    # fp.figA(u_data, u_predicted, savingdata)

    #Plotting FIG B
    fp.figB(u_data, u_predicted, savingdata)

    # Plotting FIG D 
    # for i in range(1,2):
    #     # delay = i
    #     fp.figD(u_data, u_predicted, savingdata, i)

    # Plotting 3D - 1B
    # fp.fig1B_3D(u_data, u_predicted, savingdata)

    # Plotting FigX
    # fp.figX(u_data, u_predicted, savingdata)

    print("Done")
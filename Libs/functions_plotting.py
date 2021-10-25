# Importing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
# import seaborn as sns; sns.set()
import Libs.GLOBAL as G

#File-specific setup
S=0
plot_len1 = 1000
plot_len2=2999
plot_len3=50000
title = 'RUN {ctr}, dim={dim}, density={dt}, d={d}, delay={D}'.format(ctr=G.CTR, dim=G.DM, dt=G.DENSITY, d=G.d, D=G.DELAY)


def figF(axes, figure, u_actual, u_predicted, savingdata, rows, cols=1):
     # plot_len1 = p1
     for j in range(cols+rows-1):
          col = ['cornflowerblue','forestgreen','indianred','orange' ]
          axes[j].plot(range(S,S+plot_len1),u_predicted[S:S+plot_len1,j],lw=0.75,color=col[2], label = j) 
          axes[j].plot(range(S,S+plot_len1),u_actual[S:S+plot_len1, j],lw=0.75,color=col[0], label= j)
          # axes[2*i+1].set_title('Actual',fontsize=3)
          axes[j].set_ylabel("",fontsize=2)
          axes[j].set_xlabel("", fontsize=2)
          if (j != cols+rows-1):
               axes[j].xaxis.set_ticklabels([])
          axes[j].yaxis.set_ticklabels([])

     axes[0].set_title('Predicted(Red) vs Actual(Blue)',fontsize=6)
     figure.suptitle(title)

     if savingdata:
          plt.savefig('Figs/Fig-2F{c}.png'.format(c=G.CTR),dpi=600)
     plt.show()

def figC(ax1, ax0, figure, specs, u_actual, u_predicted, savingdata):
     r = min(u_predicted.shape[1], 4)
     for i in range(r):
          col = ['cornflowerblue','forestgreen','indianred','orange' ]
          ax1.plot(range(S,S+plot_len1),u_predicted[S:S+plot_len1,i],lw=0.75,color=col[i], label = i) 
          ax0.plot(range(S,S+plot_len1),u_actual[S:S+plot_len1, i],lw=0.75,color=col[i], label=i)
     ax1.set_title('Predicted',fontsize=5)
     ax0.set_title('Actual',fontsize=5)

     figure.suptitle(title)

     # if savingdata:
     #      plt.savefig('Figs/Fig-2C{c}.png'.format(c=G.CTR),dpi=600)
     plt.show()

def figA(u_data, u_predicted, savingdata):

     plt.rcParams['figure.figsize'] = [16,8]
     spec = gridspec.GridSpec( nrows=1,ncols=2,
                              wspace=0, width_ratios=[1, 1])
     fig = plt.figure()
     ax0 = fig.add_subplot(spec[1])

     pl_len3=5000
     ax0.plot(u_predicted[:pl_len3,0], u_predicted[:pl_len3,1],'ko',markersize=0.1,color='blue')
     ax0.set_xlabel(r'$u_x$',fontsize=25)
     ax0.set_ylabel("",fontsize=25)

     ax0.xaxis.set_ticklabels([])
     ax0.yaxis.set_ticklabels([])
     ax0.set_ylim([-0.2,0.2])

     ax1 = fig.add_subplot(spec[0])
     ax1.plot(u_data[:pl_len3,0], u_data[:pl_len3,1],'ko',markersize=0.1,color='red')
     ax1.set_xlabel(r'$u_x$',fontsize=25)
     ax1.set_ylabel(r'$u_y$',fontsize=25)
     ax1.xaxis.set_ticklabels([])
     ax1.yaxis.set_ticklabels([])
     ax1.set_ylim([-0.2,0.2])

     fig.suptitle(title)

     if savingdata:
          plt.savefig('Figs/Fig-2A{c}.png'.format(c=G.CTR),dpi=600)
     plt.show()

def figB(u_data, u_predicted, savingdata): 
     plt.rcParams['figure.figsize'] = [15, 4]
     spec = gridspec.GridSpec( nrows=4,ncols=1, hspace=0.2,height_ratios=[0.25,0.25, 0.25, 0.25])
     _rows = 4
     _cols = 1
     fig=plt.figure()
     axes = []
     for i in range(_cols):
          for j in range(_rows):
               axes.append(fig.add_subplot(spec[j,i]))

     labels = ["x_pt1", "y_pt1", "x_pt2", "y_pt2"]
     for j in range(_rows*_cols):
          a = u_data[:,j]
          p = u_predicted[:,j]
          axes[j].hist(a, color='blue', alpha=0.75, label = 'actual')
          axes[j].hist(p, color='orangered', alpha = 0.75, label='predicted')
          
          axes[j].set_ylabel(labels[j], rotation = 0, fontsize = 7)
          axes[j].set_xlabel("")
          if (j!=(_rows*_cols-1)):
               axes[j].xaxis.set_ticklabels([])
               axes[0].legend(loc='upper left', prop={'size': 4})
          axes[j].yaxis.set_ticklabels([])

     # fig.suptitle("Title for whole figure", fontsize=16)
     axes[0].set_title(title,fontsize=16)
     

     if savingdata:
          plt.savefig('Figs/Fig-2B{c}.png'.format(c=G.CTR),dpi=600)
     plt.show()

def figD(u_data, u_predicted, savingdata, d = 1):
     plt.rcParams['figure.figsize'] = [16, 8]
     spec = gridspec.GridSpec( nrows=1,ncols=2,
                              wspace=0, width_ratios=[1, 1])
     fig = plt.figure()
     ax0 = fig.add_subplot(spec[1])

     delay = d

     pl_len3=min(4000, u_data.shape[0])
     ax0.plot(u_predicted[0:pl_len3-delay], u_predicted[delay:pl_len3],'ko',markersize=0.05,color='blue')
     ax0.set_xlabel(r'$u_{n}$',fontsize=25)
     ax0.set_ylabel("",fontsize=25)

     ax0.xaxis.set_ticklabels([])
     ax0.yaxis.set_ticklabels([])
     ax0.set_ylim([-0.5,0.5])

     ax1 = fig.add_subplot(spec[0])
     ax1.plot(u_data[0:pl_len3-delay], u_data[delay:pl_len3],'ko',markersize=0.1,color='red')
     ax1.set_xlabel(r'$u_{n}$',fontsize=25)
     ax1.set_ylabel(r'$u_{n+d}$',fontsize=25)
     ax1.xaxis.set_ticklabels([])
     ax1.yaxis.set_ticklabels([])
     ax1.set_ylim([-0.5,0.5])

     fig.suptitle(title)
     if savingdata:
          plt.savefig('Figs/Fig-2D.png',dpi=600)
     plt.show()

def fig1B_3D(u_data, u_predicted, savingdata):
     plt.rcParams['figure.figsize'] = [14, 14]
     plt.rcParams['agg.path.chunksize'] = 20000
     fig = plt.figure()
     ax0 = fig.add_subplot(projection='3d')

     pl_len3=50000
     ax0.plot(u_data[:pl_len3,0], u_data[:pl_len3,1],u_data[:pl_len3,2],lw=0.25,color='red',label='Actual')
     ax0.plot(u_predicted[:pl_len3,0], u_predicted[:pl_len3,1],u_predicted[:pl_len3,2],lw=0.25,color='darkblue',label='Predicted')
     ax0.set_xlabel("x")
     ax0.set_ylabel("y")
     ax0.set_zlabel("z")
     ax0.xaxis.set_ticklabels([])
     ax0.yaxis.set_ticklabels([])
     ax0.zaxis.set_ticklabels([])
     ax0.set_facecolor('white')
     # ax0.axis('off')

     h2 = Line2D([0], [0], marker='o', markersize=2, color='darkblue', linestyle='None')
     h1 = Line2D([0], [0], marker='o', markersize=2, color='r', linestyle='None')

     plt.legend([h1, h2], ['Actual', 'Predicted'], loc="upper left", markerscale=2,
               scatterpoints=1, fontsize=10,bbox_to_anchor=(0.1,0.7))

     fig.suptitle(title)
     # if savingdata:
     #      plt.savefig('Figs/Fig-1B_3D{c}.png'.format(c=G.CTR),dpi=600)
     plt.show()

def figX(u_data, u_predicted, savingdata):
     a=np.zeros(np.minimum(u_predicted.shape[0],u_data.shape[0]))
     aa=np.zeros(np.minimum(u_predicted.shape[0],u_data.shape[0]))
     
     plt.rcParams['figure.figsize'] = [15, 5]
     fig,ax=plt.subplots()
     ax.plot(a[0:300],color='darkblue',lw=0.8,alpha=0.5)
     ax.plot(aa[0:300],color='red',lw=0.8,alpha=0.5)
     h2 = Line2D([0], [0], marker='o', markersize=2, color='darkblue', linestyle='None')
     h1 = Line2D([0], [0], marker='o', markersize=2, color='r', linestyle='None')

     plt.legend([h1, h2], ['Actual', 'Predicted'], loc="lower left", markerscale=2,
               scatterpoints=1, fontsize=13)

     if savingdata:
          plt.savefig('Figs/Fig-X{c}.png'.format(c=G.CTR),dpi=600)
     plt.show()


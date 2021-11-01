import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns; sns.set()

# dbl_pend=np.loadtxt('/home/jo-anne/Documents/Honours/double-pendulum-chaotic/Data/dp_training_reworked/dp{c}.txt'.format(c=ctr))
# dbl_pend = dbl_pend[:,2:]

colours = ['orange', 'cornflowerblue','forestgreen','indianred', 'grey' ]

def somestats(dbl_pend, title=''):
    raw_mean =  np.mean(dbl_pend,axis=0)
    raw_std = np.std(dbl_pend, axis=0)
    raw_absmax = np.max(np.abs(dbl_pend), axis=0)
    raw_max = np.max(dbl_pend, axis=0)

    if title != '':
        print('==============',str.upper(title),'====================')
    print('raw_mean:', np.round(raw_mean,5))
    print('raw_max:',np.round(raw_max,5) )
    print("raw_absmax:", np.round(raw_absmax, 5))
    print('pulled_std:',np.round( raw_std,5))


def fig2D(dbl_pend, ctr, r=[]):
    ### SETUP 
    plt.rcParams['figure.figsize'] = [15, 4]
    spec = gridspec.GridSpec( nrows=4,ncols=1, hspace=1,height_ratios=[0.75,0.75, 0.75, 0.75])

    rows = 4
    cols = 1
    fig=plt.figure()
    axes = []
    for i in range(cols):
        for j in range(rows):
            axes.append(fig.add_subplot(spec[j,i]))

    if r==[]:
        S=0
        plot_len1 = dbl_pend.shape[0]
    else:
        S= r[0]
        plot_len1 = r[1]
    for j in range(cols+rows-1):
        axes[0].plot(range(S,S+plot_len1),dbl_pend[S:S+plot_len1,j],lw=0.75,color=colours[j], label = j) 
        # axes[j].set_ylabel("",fontsize=2)
        if (j != cols+rows-2):
            # axes[j].xaxis.set_ticklabels([])
            axes[j].xaxis.get_label().set_fontsize(2)
            # axes[j].set_xlabel("", fontsize=2)
        # axes[j].yaxis.set_ticklabels([])

    axes[0].set_title('dp{i}'.format(i=ctr),fontsize=6)
    fig.suptitle('Visualise Data')
    # plt.savefig('Figs/Original/{c}_shape_{s}.png'.format(c=ctr, s=dbl_pend.shape[0]),dpi=600)
    plt.show()

def fig3D(dbl_pend, ctr):
    cols = [1,3]
    coords = ['y','y']
    for column in cols:    
        from matplotlib.lines import Line2D
        plt.rcParams['figure.figsize'] = [14, 14]
        plt.rcParams['agg.path.chunksize'] = 20000
        fig = plt.figure()
        ax0 = fig.add_subplot(projection='3d')

        pl_len3=dbl_pend.shape[0]
        d = [20,40,60, 100]
        for i in range(len(d)):
            delay = d[i]
            c = colours[i]
        # ax0.plot(dbl_pend[:pl_len3,0], dbl_pend[:pl_len3,1],dbl_pend[:pl_len3,2],lw=0.25,color='forestgreen',label='Actual')
            ax0.plot(dbl_pend[0:pl_len3-2*delay,column], dbl_pend[delay:pl_len3-delay,column],dbl_pend[2*delay:pl_len3,column],lw=0.25,color=c,label='Delay {d}'.format(d=delay))
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_zlabel("z")
        ax0.xaxis.set_ticklabels([])
        ax0.yaxis.set_ticklabels([])
        ax0.zaxis.set_ticklabels([])
        ax0.set_facecolor('white')
        # ax0.axis('off')

        h = [Line2D([0], [0], marker='o', markersize=2, color=colours[i], linestyle='None') for i in range(len(d))]

        plt.legend(h, [d[i] for i in range(len(d))], loc="upper left", markerscale=2,
                scatterpoints=1, fontsize=10,bbox_to_anchor=(0.1,0.7))

        fig.suptitle('dp{i}{j}'.format(i=ctr,j=coords[0]),fontsize=6)
        plt.show()

def hist(dbl_pend, ctr, title, new_dbl_pend=''):
    ### SETUP 
    # plt.rcParams['figure.figsize'] = [15, 4]
    spec = gridspec.GridSpec( nrows=4,ncols=1, hspace=1,height_ratios=[1,1, 1, 1])

    rows = 4
    cols = 1
    fig=plt.figure()
    axes = []
    for i in range(cols):
        for j in range(rows):
            axes.append(fig.add_subplot(spec[j,i]))

    S=0
    plot_len1 = dbl_pend.shape[0]
    for j in range(cols+rows-1):
        axes[j].hist(dbl_pend[S:S+plot_len1,j],color=colours[j], alpha=0.5, label = j) 
        if new_dbl_pend!='':
            plot_len1 = new_dbl_pend.shape[0]
            axes[j].hist(new_dbl_pend[S:S+plot_len1,j],color='grey', alpha=0.5, label = j) 
        # axes[j].set_ylabel("",fontsize=2)
        if (j != cols+rows-2):
            # axes[j].xaxis.set_ticklabels(fontsize=3)
            axes[j].xaxis.get_label().set_fontsize(2)
            # axes[j].set_xlabel("", fontsize=2)
        # axes[j].yaxis.set_ticklabels([])


    axes[0].set_title('{t}_dp{i}'.format(t=title, i=ctr),fontsize=6)
    fig.suptitle('Visualise Data')

    plt.show()

def createShapesTextFile():
    shapes = []
    for i in range(40):
        dbl_pend=np.loadtxt('/home/jo-anne/Documents/Honours/double-pendulum-chaotic/Data/dp_training_reworked/dp{c}.txt'.format(c=i))
        tuple = [i, dbl_pend.shape[0]]
        shapes.append(tuple)
    np.savetxt('shapes.txt', shapes , fmt='%i', delimiter = '\t')

# for ctr in range(40):
#     dbl_pend=np.loadtxt('/home/jo-anne/Documents/Honours/double-pendulum-chaotic/Data/dp_training_reworked/dp{c}.txt'.format(c=ctr))
#     dbl_pend = dbl_pend[:,2:]
#     fig2D(ctr)
#     # hist()
#     # fig3D()
    

# dp_a = np.loadtxt('Data/actual_dp6_8.txt')[0:1000,:]
dp_p = np.loadtxt('Data/distances_b1b45b88.txt')
dp_check= np.loadtxt('Data/distances_b1b88.txt')
print(dp_p.shape, dp_check.shape, sep='\n')
# somestats(dp_a, 'actual')
somestats(dp_p, 'pred')
somestats(dp_check, 'check')

# hist(dp_a, 6, 'actual', dp_p)



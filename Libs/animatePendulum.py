#import libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import Libs.functions_creating as funcs
import sys
import Libs.GLOBAL as G
# import ffmpeg

S=G.TL + G.DC
#loadData
true = np.loadtxt('/home/jo-anne/Documents/Honours/pendulum/Data/1Coord_comp_actual.txt')
predicted = np.loadtxt('/home/jo-anne/Documents/Honours/pendulum/Data/1Coord_comp_predicted.txt')

#print before
print('\nBEFORE\nmax: ', np.max(true, axis=0), '\nmean: ', np.mean(true, axis=0), '\nstd.dev: ', np.std(true, axis=0))
print('\nBEFORE\nmax: ', np.max(predicted, axis=0), '\nmean: ', np.mean(predicted, axis=0), '\nstd.dev: ', np.std(predicted, axis=0))

#recreate predicted data - (unnormalise data)
recreate = np.loadtxt('Data/recreate.txt')
G.raw_mean = recreate[0]
G.raw_max = recreate[1]
G.raw_std = recreate[2]


x = (np.multiply(G.raw_max, G.raw_std)/0.5)
true = true * x.T + G.raw_mean
predicted = predicted * x.T + G.raw_mean
# true = ((G.raw_max * G.raw_std * true)/0.5) + G.raw_mean

#print after
print('\nAFTER\nmax: ', np.max(true, axis=0), '\nmean: ', np.mean(true, axis=0), '\nstd.dev: ', np.std(true, axis=0))
print('\nAFTER\nmax: ', np.max(predicted, axis=0), '\nmean: ', np.mean(predicted, axis=0), '\nstd.dev: ', np.std(predicted, axis=0))

#Shorten Data
end = true.shape[0]
begin = 1000
true = true[begin:end]
predicted = predicted[begin:end]

# # make lengths smaller
true = true/5.0
predicted = predicted/5.0

print(true.shape, '\n', predicted.shape, sep = '')

# Set up animation
origin = [0,0]
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1, 1), ylim=(-1,1))
# ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-3, 3), ylim=(-3,3))
ax.grid()
lineR, = ax.plot([], [], 'r-o', lw=2)
lineE, = ax.plot([], [], 'g-o', lw=2)
#line, = ax.plot([], [], 'k-', lw=1)

ax.legend(handles=[lineR, lineE], labels=["actual", "predicted"])
ax.set_title("Actual vs Predicted")


def constructCors(i, p1, p2):
    # Construct coordinates
    x = (origin[0], p1[i][0], p2[i][0])
    y = (origin[1], p1[i][1], p2[i][1])
    return (x,y)

def init():
    #initialize animation
    lineR.set_data([], [])
    lineE.set_data([], [])
    return lineR, lineE, 

def animate(i):
    #perform animation step
    lineR.set_data(*constructCors(i,true[:,:2],true[:,2:4]))
    lineE.set_data(*constructCors(i,predicted[:,:2],predicted[:,2:4]))
    
    return lineR, lineE, #line,

plt.style.use('seaborn-deep')

ani = animation.FuncAnimation(fig, animate, np.arange(1,min(true.shape[0],predicted.shape[0])), 
                            interval=1, blit=True, init_func=init)

fig.show()


i = input("Press Enter to Finish\n")
print("End of drawPendulumData.py")

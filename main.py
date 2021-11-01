import Libs.GLOBAL as G

#Initialise
G.doPCA = False #input('Do PCA-component analysis?\n')== ('y' or 'Y')
G.CTR = input("Ctr value: \n") 

#Run Simulation
file = 'Data/cor_vel_for_b1b88.txt'
#import Libs.predictor as predictor
#predictor.preds(file)
#import Libs.plotter as plotter
#plotter.plots()

#Save Data
print("Saving variable data to text file")

f = open("Libs/GLOBAL.py","r")
lines = f.readlines()
lines.append("FILE = {d}".format(d=file))
lines.append(input("Enter extra info:\n"))

from numpy import savetxt
savetxt("TextFiles/molecular_dynamics/mostRecentVariables{i}.txt".format(i=G.CTR), lines, fmt='%s')



import Libs.GLOBAL as G

#Initialise
G.doPCA = False #input('Do PCA-component analysis?\n')== ('y' or 'Y')
G.CTR = input("Ctr value: \n") 

#Run Simulation
dp = "dp6"
import Libs.predictor as predictor
predictor.preds(dp)
import Libs.plotter as plotter
plotter.plots(dp)

#Save Data
print("Saving variable data to text file")

f = open("Libs/GLOBAL.py","r")
lines = f.readlines()
lines.append("FILE = {d}".format(d=dp))
lines.append(input("Enter extra info:\n"))

from numpy import savetxt
savetxt("TextFiles/mostRecentVariables{i}.txt".format(i=G.CTR), lines, fmt='%s')



#RESET VALUES
raw_mean = 0
raw_std = 0
raw_max = []

#Initial Conditions for numerical approximation
N = 19000                                                         #number predicted
# origin = [0,0]

ALPHA = 0.99
A = 0.5

#Training Setup
TL= 8500  #N//2                                        #train_len= Number of datasteps used in training
PL= 1000   #N//2                                         #prediction_len= Timesteps predicted into the future after training
DC=250                                                          #discard=Wait for the Network to forget
DM=96                                                          #dim= dimension

#Trainer Input
HIDDEN_LAYERS = 16
LAYER_DIM = 64
EPOCHS = 128#256
BATCH_SIZE = 128

#SDD, MDD, AMDD
DELAY = 30
d = 1
doPCA = False
numPCA = 100

#OTHER
DENSITY = 0.1
verbosity = True
SPARSE_IMPLEMENTATION = False
CTR = -1

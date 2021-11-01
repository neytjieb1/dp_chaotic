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
TL= 500                                                   #train_len= Number of datasteps used in training
PL= 50                                                     #prediction_len= Timesteps predicted into the future after training
DC=50                                                          #discard=Wait for the Network to forget
DM=24                                                         #dim= dimension

#Trainer Input
HIDDEN_LAYERS = 8
LAYER_DIM = 12
EPOCHS = 16#128#128#256
BATCH_SIZE = 8#128

#SDD, MDD, AMDD
DELAY = 5 #30
d = 1
PCA = False
numPCA = 100

#OTHER
DENSITY = 0.1
verbosity = True
SPARSE_IMPLEMENTATION = False
CTR = -1


import numpy as np
from tqdm import tqdm
import os


def scaleData(coords, scalefactor):
    coords = coords/scalefactor

    #Translate
    #x
    cols = [0,2,4]
    coords[:, cols] -= coords[0,0]
    #y
    cols = [1,3,5]
    coords[:, cols] -= coords[0,1]

    #Rotate
    c = np.zeros((coords.shape[0], coords.shape[1]))
    cols = [0,2,4]
    #insert x
    for i in cols:
        c[:,i+1] = -coords[:,i]
    #insert y
    cols = [1,3,5]
    for j in cols:
        c[:,j-1] = coords[:,j]

    return c



#some stuff
train_dir = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_csv/train'


# load in all separate files
for filename in tqdm([x for x in os.listdir(train_dir) if not x.startswith('.')]):
    csv_file = os.path.join(train_dir, filename)
    coords = np.genfromtxt(csv_file,delimiter=' ')
    # change files
    coords = scaleData(coords, 200)
    
    #save files
    sep = '.'
    stripped = int(filename.split(sep, 1)[0])
    np.savetxt('dp_training_reworked/dp{i}.txt'.format(i=stripped), coords)



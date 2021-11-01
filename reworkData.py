import numpy as np

coord = np.loadtxt('Data/fullcoords.txt')
veloc = np.loadtxt('Data/fullvel.txt')
force = np.loadtxt('Data/fullforce.txt')

coord = np.append(coord[:,0:3], coord[:,88*3:89*3], axis=1)
veloc = np.append(veloc[:,0:3], veloc[:,88*3:89*3], axis=1)
force = np.append(force[:,0:3], force[:,88*3:89*3], axis=1)

total = np.append(coord, veloc, axis=1)
total = np.append(total, force, axis=1)

print(coord.shape, veloc.shape, force.shape, total.shape, sep='\n')
np.savetxt('Data/cor_vel_for_b1b88.txt', total)

#  For DP_CHAOTIC
# from tqdm import tqdm
# import os
# def scaleData(coords, scalefactor):
#     coords = coords/scalefactor

#     #Translate
#     #x
#     cols = [0,2,4]
#     coords[:, cols] -= coords[0,0]
#     #y
#     cols = [1,3,5]
#     coords[:, cols] -= coords[0,1]

#     #Rotate
#     c = np.zeros((coords.shape[0], coords.shape[1]))
#     cols = [0,2,4]
#     #insert x
#     for i in cols:
#         c[:,i+1] = -coords[:,i]
#     #insert y
#     cols = [1,3,5]
#     for j in cols:
#         c[:,j-1] = coords[:,j]

#     return c

# #some stuff
# train_dir = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_csv/train'

# # load in all separate files
# for filename in tqdm([x for x in os.listdir(train_dir) if not x.startswith('.')]):
#     csv_file = os.path.join(train_dir, filename)
#     coords = np.genfromtxt(csv_file,delimiter=' ')
#     # change files
#     coords = scaleData(coords, 200)
    
#     #save files
#     sep = '.'
#     stripped = int(filename.split(sep, 1)[0])
#     np.savetxt('dp_training_reworked/dp{i}.txt'.format(i=stripped), coords)



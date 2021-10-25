import numpy as np
import Libs.GLOBAL as G

def normalise_data(_data, MD):

    raw_data = _data
    
    G.raw_mean = np.mean(raw_data,axis=0)
    G.raw_std = np.std(raw_data, axis=0) #1
    data = ( raw_data - G.raw_mean ) / G.raw_std
    G.raw_max = np.max(np.abs(data), axis=0)
    
    if G.verbosity:
        print('raw_mean:', G.raw_mean)
        print('pulled_mean:', np.round(np.mean(data, axis=0)))
        print('pulled_std:', np.std(data, axis=0))
        print("current max:", G.raw_max)
    
    if MD:
        recreate = np.zeros(3)      # if MD
    else:
        recreate = np.zeros((3,data.shape[1]))                    #in the case where working with angles and want to work backwards
    recreate[0] = G.raw_mean
    recreate[1] = G.raw_max
    recreate[2] = G.raw_std
    np.savetxt('Data/recreate.txt', recreate)

    # data = ((raw-mean)/std)*(0.5/max)
    # raw = ((max*data*std)/0.5)+mean

    data = data*(0.5/G.raw_max)

    if G.verbosity:
        print('\nmax: ', np.max(data, axis=0), '\nmean: ', np.round(np.mean(data, axis=0),4), '\nstd: ', np.std(data, axis=0))
    
    return data


def createDelCoordData(_data, MD):
    raw_data = normalise_data(_data, MD)

    G.verbosity = False                         #False to avoid vectorised output for 1D
    d = G.d

    u = np.zeros((raw_data.shape[0]-d, d))
    for i in range(raw_data.shape[0]-d):
        temp = np.zeros(d)
        for t in range(d):
            # print(temp[t].shape, raw_data[t+1].shape)
            temp[t] = raw_data[t+i]

        # print(u[i].shape, "\t", temp.shape  )
        u[i] = temp
    
    return u

def combine_coords(_data):
    gam = 0.1
    data = np.zeros((_data.shape[0], 2))
    data[:,0]=1/10*(np.sin(gam*_data[:,0])+np.sin(gam*_data[:,1]))
    data[:,1]=1/10*(np.sin(gam*_data[:,2])+np.sin(gam*_data[:,3]))

    return data
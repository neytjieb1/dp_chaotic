import numpy as np
import seaborn as sns; sns.set()
import Libs.GLOBAL as G
# from Libs.normalise_delay_data import normalise_data



def normalise_data(dp):

    if G.verbosity:
        print('Normalising Data...\n')  
    raw_data = np.loadtxt('/home/jo-anne/Documents/Honours/double-pendulum-chaotic/Data/dp_training_reworked/{d}.txt'.format(d=dp))
    raw_data = raw_data[:,2:]   #get rid of origin

    # raw_data = np.reshape(raw_data, (19000,1))  #if only distances
    
    G.raw_mean = np.mean(raw_data,axis=0)
    G.raw_std = np.std(raw_data, axis=0)
    data = ( raw_data - G.raw_mean ) / G.raw_std
    G.raw_max = np.max(np.abs(data), axis=0)
    
    if G.verbosity:
        print('raw_mean:', np.round(G.raw_mean,5))
        print('\npulled_mean:',np.round(np.mean(data, axis=0),5) )
        print('\npulled_std:',np.round( np.std(data, axis=0),5))
        print("current max:", G.raw_max)
    
    recreate = np.zeros((3,raw_data.shape[1]))                    #in the case where working with angles and want to work backwards
    recreate[0] = G.raw_mean
    recreate[1] = G.raw_max
    recreate[2] = G.raw_std
    np.savetxt('Data/recreate.txt', recreate)

    data = data*(0.5/G.raw_max)

    if G.verbosity:
        print('\nmax: ', np.max(data, axis=0), '\nmean: ', np.round(np.mean(data, axis=0),4), '\nstd: ', np.std(data, axis=0))
    
    return data


def preds(dp):
        
    ##### Load Data
    data=normalise_data(dp)

    u_data=data

    #############################################
    # Prediction
    #############################################

    prediction_len=G.PL                                                                                                #Timesteps predicted into the future after training
    discard=G.DC                                                                                                                #Wait for the Network to forget
    train_len= min(G.TL, u_data.shape[0]-  G.PL - G.DC)                              # Number of datasteps used in training
    dim=G.DM                                                                                                                          #Dimension of the Network

    X=np.zeros((train_len+discard+1,dim))

    rng=np.random.default_rng(12345)
    X[0]=rng.random(dim)

    from Libs.Net import Network
    print('\nImporting Net.py')
    net=Network(dim,alpha=0.99,a=0.5)           # alpha in (0.8, 1.2), a in (0,1)

    # Driving the system
    print("\nDriving the System...")
    for t in range(train_len+discard):
        print("Computed {} out of {} \r".format(t,train_len+discard),end='')    
        X[t+1]=net.g(u_data[t],X[t])
    print('')

    np.savetxt('X.txt', X)

    # Setting Up training Data: first {discard} datapoints are ignored
    Y_train=u_data[discard:discard+train_len]

    ## Constructing a stack of {delay} time delayed vectors
    delay=G.DELAY                                  
    X_train=np.zeros((train_len,delay*dim))

    for i in range(delay):
        X_train[:,i*dim:(i+1)*dim]=X[discard-delay+i+1:discard-delay+i+1+train_len]


    if G.PCA:
        print("\n\nDOING PCA. Dim of X-train = {d}".format(d=X_train.shape[1]))
        G.NUMPCAs=  int(input("Choose number of PCA-components: "))  #X_train.shape[1]//2              #dimensions*delay / 2
    #     temp = X_train[discard: discard+ train_len]
        import  scipy.linalg as LA
        U,Sig, Wt=LA.svd(X_train, full_matrices=True)
        T_X= U[:,: G.DELAY*dim]*Sig
        X_train = T_X[:,0:G.NUMPCAs]

    ### Training Gamma.
    from Libs.Train_Gamma import Train_NN_PCA
    print('\nSetting up Trainer')
    trainer=Train_NN_PCA(X_train,Y_train,hidden_layers=G.HIDDEN_LAYERS,layer_dimension=G.LAYER_DIM, epochs= G.EPOCHS, batch_size=G.BATCH_SIZE)  

    # Setting up variables
    u_predicted=np.zeros((prediction_len,u_data.shape[1]))
    u_k=u_data[train_len+discard]
    x_0=X[train_len+discard]
    u_predicted[0]=u_k
    X_stack=np.zeros(delay*dim)

    for i in range(delay):
        X_stack[i*dim:(i+1)*dim]=X[train_len+discard-delay+i+1]

    S=train_len+discard

    # Running Prediction
    for t in range(prediction_len-1):
        print("Predicted {} out of {} \r".format(t,prediction_len),end='')
        x_1=net.g(u_k,x_0)
        # X[S+t+1]=x_1

        for i in range(delay-1):
            X_stack[i*dim:(i+1)*dim]=X_stack[(i+1)*dim:(i+2)*dim]

        X_stack[(delay-1)*dim:delay*dim]=x_1
        
        u_k=trainer.predict(X_stack); 
        #print(u_k)
        u_predicted[t+1]=u_k
        x_0=x_1
        
    #############################################
    # Saving Data
    #############################################

    if G.verbosity:
        print(u_predicted.shape[0], u_data[S:].shape[0])
        print("Saving Generated data to files")

    np.savetxt('Data/predicted_{d}_{c}.txt'.format(d=dp, c=G.CTR),u_predicted)
    np.savetxt('Data/actual_{d}_{c}.txt'.format(d=dp, c=G.CTR),u_data[S:])

    if G.verbosity:
        print('Reached end of predictor.py\n')

    import os
    duration = 1.2  # seconds
    freq = 450  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))




    # if G.doPCA:
    #     G.NUMPCAs = X_train.shape[1]//2
    #     print("\nFinding PCA Vals")
    #     temp = X_train[discard: discard+ train_len]
    #     U,Sig, Wt=LA.svd(temp, full_matrices=True)
    #     T_X= U[:,: G.DELAY*dim]*Sig
    #     X_train = T_X[:,0:G.NUMPCAs]
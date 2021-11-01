"""
Class to wrap all the tensorflow/keras details for training Gamma
"""


import numpy as np 
import tensorflow as tf  ## USING v 2.0
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
        # pd.DataFrame(self.history.history).plot(figsize=(8,5))
        # plt.show()
import Libs.GLOBAL as G


class Train_NN_PCA:
    def __init__(self, X_train,Y_train, verbose=True,hidden_layers=12,layer_dimension=64,epochs=150,batch_size=128, config=0):
        l,dim_y=Y_train.shape   #trainlen
        l,dim_x=X_train.shape   #TL x delay*dim  #switched these two around
        
        self.dim_x=dim_x

        # Principal components

        # X_pca=X_train[:,0:dim_x//2]

        # U,Sig,Wt=np.linalg.svd(X_pca,full_matrices=True)
        # T=U[:,:dim_x//2]*Sig
        # self.W=Wt.T  #Principal component matrix


        # if G.PCA:
        #     l =min(X_train.shape[0], Y_train.shape[0])
        #     T_train=np.zeros((l,dim_x))
        #     T_train[:,0:(dim_x//2)]=T
        #     T_train[:,(dim_x//2):dim_x]=X_train[:,(dim_x//2):dim_x]@self.W
        # else:
        #     T_train=np.zeros((l,dim_x))
        #     T_train[:,0:(dim_x//2)]=T
        #     T_train[:,(dim_x//2):dim_x]=X_train[:,(dim_x//2):dim_x]@self.W

        T_train = X_train

        EPOCHS=epochs
        BATCH_SIZE=batch_size
        VERBOSE=verbose
        VALIDATION_SPLIT=0.33       # was 0.2

        # Defining the structure of the feed forward neural network using keras.
        tf.keras.backend.clear_session()
        self.model =tf.keras.Sequential()
        self.model.add(keras.layers.Dense(layer_dimension, input_shape=(dim_x,), name='input_layer', activation='relu'))
        for i in range(hidden_layers):
            self.model.add(keras.layers.Dense(layer_dimension, name='hidden_layer_'+str(i), activation='relu'))

        for j in range(hidden_layers//2):
            self.model.add(keras.layers.Dense(layer_dimension//2, name='hidden_layer_'+str(j+hidden_layers), activation='relu'))
            #add dropout layer to prevent overfitting
            # self.model.add(keras.layers.Dropout(0.4))
        self.model.add(keras.layers.Dense(dim_y, name='output_layer', activation='tanh'))

        if VERBOSE:
            self.model.summary()

        # Training with different learning rates for the Adam Optimizer
        
        rates = [ 0.001, 0.0001, 0.00001]#, 0.000001 ]
        self.history = 0
        for lr in rates:
            opt=keras.optimizers.Adam(learning_rate=lr)
            self.model.compile(optimizer=opt, loss='MSE')
            # self.model.compile(optimizer=opt, loss ="categorical_crossentropy", metrics =['accuracy'])

            from keras import callbacks
            earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                                    mode ="min", patience = 5, 
                                                    restore_best_weights = True)
                                                    
            self.history = self.model.fit(T_train,Y_train, batch_size=BATCH_SIZE,epochs=EPOCHS, 
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
            # self.plotModelTraining()
        
        # import pandas as pd
        # pd.DataFrame(self.history.history).plot(figsize=(8,5))
        # plt.show()

        
    def plotModelTraining(self):
        print(self.history.history.keys())
        #  "Accuracy"
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    

    def predict(self,x):
        dim_x=self.dim_x
        
        if G.PCA:
            x_ = np.zeros((1, dim_x*2))
        else:
            x_=np.zeros((1,dim_x))      #dim_x = delay*dim

        x_[0]=x
        # t_=np.zeros((1,dim_x))
        # t_[0,0:(dim_x//2)]=x_[0,0:(dim_x//2)]@self.W
        # t_[0,(dim_x//2):dim_x]=x_[0,(dim_x//2):dim_x]@self.W; 
        # np.savetxt('t_.txt', t_)

        # return self.model.predict(t_)[0]
        return self.model.predict(x_)[0]



    def principal_components(self,X_array):
        return X_array@self.W

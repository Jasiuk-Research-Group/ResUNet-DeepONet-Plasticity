import sys
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from deepxde.backend import tf
import keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import math
import tensorflow as tf
import time as TT
tf.config.set_soft_device_placement(True)

########################################################################################################
seed = 2023 
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(seed)

# Parameters
m = 128*128 # Number of geometry parameters
N_load_paras = 2 # Number of load parameters
fraction_train = 0.8
HIDDEN = 256
data_type = np.float32
sub = '_Vanilla_DON_HD256_LR5e-4'

print('\n\nModel parameters:')
print( sub )
print( 'fraction_train  ' , fraction_train )
print('\n\n\n')


# Branch network, for encoding load information
my_act = "relu"
branch = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(2+128*128,)),
        tf.keras.layers.Dense(175, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'), 
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'),
        #Dropout( drop_rate ),
        tf.keras.layers.Dense( HIDDEN, activation=my_act , kernel_initializer='GlorotNormal'),
                              ]
)
print('\n\nBranch network:')
branch.summary()


# Trunk network, for encoding the geometry
trunk = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'), 
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(256, activation=my_act, kernel_initializer='GlorotNormal'),
        #Dropout( drop_rate ),
        tf.keras.layers.Dense( HIDDEN, activation=my_act , kernel_initializer='GlorotNormal'),
                              ]
)
print('\n\nTrunk network:')
trunk.summary()
# exit()

# Build DeepONet
activation = "relu"
net = dde.maps.DeepONetCartesianProd(
        [ m+2 , branch ], [2, 256, 256, 256, 256, 256, 256] , activation, "Glorot normal")


# Load raw data
nnn = 3000
# Geometry
base0 = './Data/Elastic_TO_128x128/'
tmp = np.load(base0+'FinalDesigns_combined.npz')['data']
data_raw = np.zeros([ nnn * 5 , 128**2 ])
for i in range( nnn ):
    data_raw[ 5*i : 5*(i+1) , : ] = tmp[i,:]

# Load parameters
base = './Data/Abaqus_128_random_load/'
LoadInfo = np.load(base+'LoadInfo.npz')['data'][ :nnn , :5 , : ].reshape([ nnn*5 , 2 ])

# Simulation results
sim_raw1 = np.load(base+'Svm_PEEQ-pt1.npz')['data']
tmp1 = np.load(base+'FailedSims-pt1.npy').flatten()
sim_raw1 = np.delete( sim_raw1 , tmp1 , axis=1 )

sim_raw2 = np.load(base+'Svm_PEEQ-pt2.npz')['data']
# tmp2 = np.load(base+'FailedSims-pt2.npy').flatten()
# sim_raw2 = np.delete( sim_raw2 , tmp2 , axis=1 )

sim_raw = np.concatenate( [sim_raw1,sim_raw2] , axis=1 )

# sim_raw = sim_raw1
stress , peeq = sim_raw[ 0 , :: ] , sim_raw[ 1 , :: ]

failed = []
for _ in tmp1:
    failed += [ _ ]
# for _ in tmp2:
#     failed += [ _ + 1500*5 ]
LoadInfo = np.delete( LoadInfo , failed , axis=0 )
data_raw = np.delete( data_raw , failed , axis=0 )


print( data_raw.shape )
print( LoadInfo.shape )
print( sim_raw.shape )


# Cap and scale
smax = 500.
stress[ stress > smax ] = smax
stress /= smax

LoadInfo[:,0] = ( LoadInfo[:,0] - 2. ) / 6. # Map to 0 - 1
LoadInfo[:,1] = ( LoadInfo[:,1] ) / np.pi # Map to 0 - 1


# Combine both inputs
tmp = np.zeros([len(data_raw),m+2])
tmp[:,:m] = data_raw
tmp[:,m:] = LoadInfo
data_raw = tmp.copy()


# Train / test split
N_valid_case = len(stress)
N_train = int( N_valid_case * fraction_train )
train_case = np.random.choice( N_valid_case , N_train , replace=False )
test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )


# Input: geometry and load info
u0_train = data_raw[ train_case , :: ].astype(np.float32)
u0_testing = data_raw[ test_case , :: ].astype(np.float32)

# Output: stress
s_train = stress[ train_case , : ].astype(data_type)
s_testing = stress[ test_case , : ].astype(data_type)

# 2D coordinates of all points
nele = 128
xy_train_testing = np.array( np.unravel_index( np.arange( nele**2 ) , [nele,nele] ) ).transpose() / float( nele - 1. )

print('u0_train.shape = ',u0_train.shape)
print('u0_testing.shape = ',u0_testing.shape)
print('s_train.shape = ',s_train.shape)
print('s_testing.shape = ',s_testing.shape)
print('xy_train_testing.shape', xy_train_testing.shape)

###################################################################################
s0_plot = s_train.flatten()
s1_plot = s_testing.flatten()
plt.hist( s0_plot , bins=50 , color='r' , alpha=0.6 , density=True ) 
plt.hist( s1_plot , bins=50 , color='b' , alpha=0.6 , density=True ) 
plt.legend(['Training' , 'Testing'])
plt.savefig('train_test_stress_dist.pdf')
plt.close()
###################################################################################


x_train = (u0_train.astype(data_type), xy_train_testing.astype(data_type))
y_train = s_train.astype(data_type) 
x_test = (u0_testing.astype(data_type), xy_train_testing.astype(data_type))
y_test = s_testing.astype(data_type)


data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
model = dde.Model(data, net)


def metric( y_train , y_pred ):
    true_vals = y_train
    pred_vals = y_pred

    flag = ( true_vals < 1e-8 )
    pred_vals[ flag ] = 0

    err = []
    for i in range(len(true_vals)):
        error_s_tmp = np.linalg.norm(true_vals[i] - pred_vals[i] ) / np.linalg.norm( true_vals[i] )
        err.append( error_s_tmp )
    return np.mean( err )

def relativeDiff( y_true, y_pred ):
    diff = y_true - y_pred
    y_true_f = K.flatten(y_true)
    diff_f = K.flatten(diff)
    return tf.norm( diff ) / tf.norm( y_true_f )


model.compile(
    "adam",
    lr=5e-4,
    decay=("inverse time", 1, 1e-4),
    # loss=relativeDiff,
    metrics=[ "mean l2 relative error" , metric ],
)
losshistory, train_state = model.train(epochs=400000, batch_size=16)
np.save('losshistory.npy',losshistory)

y_pred = model.predict(data.test_x)
np.savez_compressed('TestData'+sub+'.npz',a=y_test,b=y_pred)

st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )

error_s = []
rho = data.test_x[0]
for i in range(len(y_test)):
    true_vals = y_test[i]
    pred_vals = y_pred[i]

    flag = ( true_vals < 1e-8 )
    pred_vals[ flag ] = 0

    error_s_tmp = np.linalg.norm(true_vals - pred_vals ) / np.linalg.norm( true_vals )

    if error_s_tmp > 1:
        error_s_tmp = 1

    error_s.append(error_s_tmp)
error_s = np.array(error_s)
print("error_s = ", error_s)
np.save( 'errors'+sub+'.npy' , error_s )

print('mean of relative L2 error of s: {:.2e}'.format( np.mean(error_s) ))
print('std of relative L2 error of s: {:.2e}'.format( np.std(error_s) ))

plt.hist( error_s.flatten() , bins=25 )
plt.savefig('Err_hist_DeepONet'+sub+'.jpg' , dpi=1000)

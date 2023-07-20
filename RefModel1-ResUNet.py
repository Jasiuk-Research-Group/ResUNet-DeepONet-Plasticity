import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import deepxde as dde
from deepxde.backend import tf
import keras.backend as K

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
import math
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import keras.backend as K
import time as TT
tf.config.set_soft_device_placement(True)

class ResBlock(Layer):
    """
    Represents the Residual Block in the ResUNet architecture.
    """
    def __init__(self, filters, strides, l2 , **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.l2 = l2

        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", use_bias=False,kernel_regularizer=l2)

        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False,kernel_regularizer=l2)

        self.conv_skip = Conv2D(filters=filters, kernel_size=1, strides=strides, padding="same", use_bias=False,kernel_regularizer=l2)
        self.bn_skip = BatchNormalization()

        self.add = Add()

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        skip = self.conv_skip(inputs)
        skip = self.bn_skip(skip, training=training)

        res = self.add([x, skip])
        return res

    def get_config(self):
        return dict(filters=self.filters, strides=self.strides, **super(ResBlock, self).get_config())

def ResUNet(input_shape , classes , filters_root, depth, drop_rate , L2_reg ):
    """
    Builds ResUNet model.
    :param input_shape: Shape of the input images (h, w, c). Note that h and w must be powers of 2.
    :param classes: Number of classes that will be predicted for each pixel. Number of classes must be higher than 1.
    :param filters_root: Number of filters in the root block.
    :param depth: Depth of the architecture. Depth must be <= min(log_2(h), log_2(w)).
    :return: Tensorflow model instance.
    """
    regularizer = tf.keras.regularizers.L2(L2_reg)

    if not math.log(input_shape[0], 2).is_integer() or not math.log(input_shape[1], 2):
        raise ValueError(f"Input height ({input_shape[0]}) and width ({input_shape[1]}) must be power of two.")
    if 2 ** depth > min(input_shape[0], input_shape[1]):
        raise ValueError(f"Model has insufficient height ({input_shape[0]}) and width ({input_shape[1]}) compared to its desired depth ({depth}).")

    input = Input(shape=(input_shape[0]*input_shape[1],3) )

    layer = tf.keras.layers.Reshape( (input_shape[0],input_shape[1],3 ) )(input)


    # ENCODER
    encoder_blocks = []

    filters = filters_root
    layer = Conv2D(filters=filters, kernel_size=7, strides=1, padding="same",kernel_regularizer=regularizer)(layer)

    branch = Conv2D(filters=filters, kernel_size=7, strides=1, padding="same", use_bias=False,kernel_regularizer=regularizer)(layer)
    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=True,kernel_regularizer=regularizer)(branch)

    layer = Add()([branch, layer])

    encoder_blocks.append(layer)

    for _ in range(depth - 1):
        filters *= 2
        layer = ResBlock(filters, strides=2,l2=regularizer)(layer)
        layer = Dropout( drop_rate )(layer)
        encoder_blocks.append(layer)

    # BRIDGE
    filters *= 2
    layer = ResBlock(filters, strides=2,l2=regularizer)(layer)

    # DECODER
    for i in range(1, depth + 1):
        filters //= 2
        skip_block_connection = encoder_blocks[-i]

        layer = UpSampling2D( interpolation="bilinear" )(layer)
        # layer = Concatenate()([layer, skip_block_connection])
        layer_a = ResBlock(filters, strides=1,l2=regularizer)(layer)
        layer_b = ResBlock(filters, strides=1,l2=regularizer)(skip_block_connection)
        layer = Add()([layer_a, layer_b])

        layer = Dropout( drop_rate )(layer)

    layer = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same",kernel_regularizer=regularizer)(layer)
    layer = Activation(activation="sigmoid")(layer)

    # layer = tf.math.reduce_sum( layer , axis=-1 , keepdims=True )

    output = tf.keras.layers.Reshape( (input_shape[0]*input_shape[1], ) )(layer)
    return Model(input, output)

class DeepONetCartesianProd(dde.maps.NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = dde.maps.activations.get(activation)

        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = dde.maps.FNN(
                layer_sizes_branch,
                activation_branch,
                kernel_initializer,
                regularization=regularization,
            )
        # if callable(layer_sizes_trunk[1]):
        #    # User-defined network
        #     self.trunk = layer_sizes_trunk[1] 
        # else:
        #     self.trunk = dde.maps.FNN(
        #     layer_sizes_trunk,
        #     self.activation_trunk,
        #     kernel_initializer,
        #     regularization=regularization,
        # )
        self.trunk = None
        self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        # Branch net to encode the input function
        x = self.branch(x_func)

        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


########################################################################################################
seed = 2023 
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(seed)

# Parameters
m = [ 128 , 128 ] # Number of input sensors
HIDDEN = 1
DEPTH = 4
FILTER = 20
data_type = np.float32

sub = '_Add_ResUNet_only_3p5mil_5e-4_LR'
print( sub )


activation = "relu"
branch = ResUNet(input_shape=(m[0], m[1], 3), classes=HIDDEN, filters_root=FILTER, depth=DEPTH , drop_rate=0.02 , L2_reg=5e-3 )
branch.summary()

net = DeepONetCartesianProd(
        [ m , branch], [2, None] , activation, "Glorot normal")
# exit()


# Load raw data
nnn = 3000
# Geometry
base0 = './Data/Elastic_TO_128x128/'
tmp = np.load(base0+'FinalDesigns_combined.npz')['data']
data_raw = np.zeros([ nnn * 5 , 128**2 , 3 ])
for i in range( nnn ):
    data_raw[ 5*i : 5*(i+1) , : , 0 ] = tmp[i,:]

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


for i in range( len(LoadInfo) ):
    data_raw[ i , : , 1 ] = LoadInfo[i,0]
    data_raw[ i , : , 2 ] = LoadInfo[i,1]

###################################################################################################################


# Train / test split
N_valid_case = len(stress)
N_train = int( N_valid_case * 0.8 )
train_case = np.random.choice( N_valid_case , N_train , replace=False )
test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )

u0_train = data_raw[ train_case , :: ].astype(data_type)
u0_testing = data_raw[ test_case , :: ].astype(data_type)


s_train = stress[ train_case , : ].astype(data_type)
s_testing = stress[ test_case , : ].astype(data_type)


###################################################################################
s0_plot = s_train.flatten()
s1_plot = s_testing.flatten()
plt.hist( s0_plot , bins=50 , color='r' , alpha=0.6 , density=True ) 
plt.hist( s1_plot , bins=50 , color='b' , alpha=0.6 , density=True ) 
plt.legend(['Training' , 'Testing'])
plt.savefig('train_test_stress_dist.pdf')
plt.close()
###################################################################################


# 2D coordinates of all points
nele = 128
xy_train_testing = np.array( np.unravel_index( np.arange( nele**2 ) , [nele,nele] ) ).transpose() / float( nele - 1. )

print('u0_train.shape = ',u0_train.shape)
print('u0_testing.shape = ',u0_testing.shape)
print('s_train.shape = ',s_train.shape)
print('s_testing.shape = ',s_testing.shape)
print('xy_train_testing.shape', xy_train_testing.shape)

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
losshistory, train_state = model.train(epochs=150000, batch_size=16)
np.save('losshistory.npy',losshistory)

st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )

np.savez_compressed('TestData'+sub+'.npz',a=y_test,b=y_pred)

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
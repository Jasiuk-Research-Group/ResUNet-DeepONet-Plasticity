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

    # Load parameters
    input2 = Input(shape=2) # load paras
    layer2 = Dense( 25 )( input2 )
    layer2 = Dense( 50 )( layer2 )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 50 )( layer2 )
    layer2 = Dense( 25 )( layer2 )
    encoded_load = Dense( 128*128 )( layer2 )


    input1 = Input(shape=input_shape[0]*input_shape[1]) # Geom
    layer = tf.keras.layers.Reshape( (input_shape[0],input_shape[1],1 ) )(input1)


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

    ##################################################################
    # This is the smallest point
    encoded_geom = tf.keras.layers.Flatten()( layer )


    ##################################################################
    # Data fusion
    mixed = tf.math.multiply( encoded_geom , encoded_load )
    layer = tf.keras.layers.Reshape( (8,8,256) )( mixed ) 

    # geometry DECODER
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
    output1 = tf.keras.layers.Reshape( (input_shape[0]*input_shape[1],classes ) )(layer)


    # Load decoder
    layer2 = Dense( 50 )( mixed )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 50 )( layer2 )
    layer2 = Dense( classes )( layer2 )
    output2 = Activation(activation="relu")(layer2)


    # output = layer
    return Model( [input1,input2] , [output1,output2] )

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

        # self.branch = layer_sizes_branch[1]
        self.trunk = layer_sizes_trunk[1] 
        self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0] # [ bs , 2 ] , load parameters
        x_loc = inputs[1] # [ bs , 128*128 ] , input geometry

        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)


        # Branch net, encode load info:
        # x_func = self.branch( [x_func] ) # Output: [ bs , hidden dim ]

        # [ bs , 128*128 , hidden dim ] [ bs , hidden dim ]
        x_loc                          , x_func            = self.trunk( [x_loc,x_func] )


        # # Trunk net, encode geometry
        # x_loc = self.activation_trunk( self.trunk(x_loc) ) # Output: [ bs , 128*128 , hidden dim ]

        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = tf.einsum("ik,ijk->ij", x_func, x_loc) # Output: [ bs , 128*128 ]

        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1][indices]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y

########################################################################################################
seed = 2023 
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(seed)

# Parameters
m = [ 128 , 128 ] # Number of geometry parameters
N_load_paras = 2 # Number of load parameters
HIDDEN = 32
DEPTH = 4
FILTER = 16
fraction_train = 0.8
data_type = np.float32
drop_rate = 0.02
L2_reg = 5e-3
sub = '_mixed_DeepONet_LR_5e-4'

print('\n\nModel parameters:')
print( sub )
print( 'HIDDEN  ' , HIDDEN )
print( 'DEPTH  ' , DEPTH )
print( 'FILTER  ' , FILTER )
print( 'fraction_train  ' , fraction_train )
print('\n\n\n')



# Trunk network, for encoding the geometry
trunk = ResUNet(input_shape=(m[0], m[1], 1), classes=HIDDEN, filters_root=FILTER, depth=DEPTH , drop_rate=drop_rate , L2_reg=L2_reg )
print('\n\nTrunk network:')
trunk.summary()


# Build DeepONet
activation = "relu"
net = DeepONetCartesianProd(
        [ N_load_paras , None ], [ m[0]*m[1] , trunk ] , activation, "Glorot normal")


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


# Train / test split
N_valid_case = len(stress)
N_train = int( N_valid_case * fraction_train )
train_case = np.random.choice( N_valid_case , N_train , replace=False )
test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )


# Branch: load parameters
u0_train = LoadInfo[ train_case , :: ].astype(data_type)
u0_testing = LoadInfo[ test_case , :: ].astype(data_type)

# Trunk: geometry
xy_train = data_raw[ train_case , :: ].astype(data_type)
xy_testing = data_raw[ test_case , :: ].astype(data_type)

# Output: stress
s_train = stress[ train_case , : ].astype(data_type)
s_testing = stress[ test_case , : ].astype(data_type)

print('u0_train.shape = ',u0_train.shape)
print('u0_testing.shape = ',u0_testing.shape)
print('xy_train.shape = ',xy_train.shape)
print('xy_testing.shape = ',xy_testing.shape)
print('s_train.shape = ',s_train.shape)
print('s_testing.shape = ',s_testing.shape)


###################################################################################
s0_plot = s_train.flatten()
s1_plot = s_testing.flatten()
plt.hist( s0_plot , bins=50 , color='r' , alpha=0.6 , density=True ) 
plt.hist( s1_plot , bins=50 , color='b' , alpha=0.6 , density=True ) 
plt.legend(['Training' , 'Testing'])
plt.savefig('train_test_stress_dist.pdf')
plt.close()
###################################################################################


# Pack data
x_train = (u0_train.astype(data_type), xy_train.astype(data_type))
y_train = s_train.astype(data_type) 
x_test = (u0_testing.astype(data_type), xy_testing.astype(data_type))
y_test = s_testing.astype(data_type)
data = TripleCartesianProd(x_train, y_train, x_test, y_test)

# Build model
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


import time as TT
st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )
np.savez_compressed('TestData'+sub+'.npz',a=y_test,b=y_pred,c=u0_testing,d=xy_testing)


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

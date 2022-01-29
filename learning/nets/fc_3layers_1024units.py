import tensorflow as tf
import learning.tf_util as TFUtil

NAME = "fc_3layers_1024units"

def build_net(input_tfs, reuse=False):
    layers = [1024, 512, 256]
    activation = tf.nn.relu

    """
    with tf.variable_scope("states"):
        h1_state = TFUtil.fc_net(input_tfs[0], [layers[0]], activation=None, reuse=reuse)

    if(len(input_tfs) > 2):
        with tf.variable_scope("others"):
            input_tf_other = tf.concat(axis=-1, values=input_tfs[1:])
            h1_other = TFUtil.fc_net(input_tf_other, [layers[0]], activation=None, reuse=reuse)
            h = TFUtil.fc_net(activation(h1_state+h1_other), [layers[1]], activation=activation, reuse=reuse)          
    else:
        h = TFUtil.fc_net(activation(h1_state), [layers[1]], activation=activation, reuse=reuse)
    """
    input_tf = tf.concat(axis=-1, values=input_tfs)
    h = TFUtil.fc_net(input_tf, layers, activation=activation, reuse=reuse)
    h = activation(h)
    return h
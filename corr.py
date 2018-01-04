
'''=============================
Correlation
============================='''

import tensorflow as tf
import tensorflow.contrib.keras as kr

def normalize_c(arr, epsilon = 1e-9):
    with tf.variable_scope('mvn'):
        mean, var = tf.nn.moments(arr, -1, keep_dims=True)
        return ( ( arr - mean ) / ( tf.sqrt(var) + epsilon ) )

def corr1d(left, right, l_disp=70, r_disp=10):
    with tf.variable_scope('correlation'):
        n, h, w, c = left.get_shape()

        corr_list = []
        for i in range(-l_disp, r_disp+1):
            offset, target = max(0, i), w-abs(i)
            crop = tf.image.crop_to_bounding_box(right, 0, offset, h, target)

            p_offset = -min(0, i)
            padding = tf.image.pad_to_bounding_box(crop, 0, p_offset, h, w)

            corr_single = tf.reduce_mean( left * padding, axis=3, keep_dims=True )
            corr_list.append(corr_single)
        corr = kr.layers.Concatenate()(corr_list)
    return corr


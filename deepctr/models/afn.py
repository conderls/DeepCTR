# -*- coding:utf-8 -*-

"""
Author:
    conderls, conderls@sina.com

Reference:
    [1] Cheng, W., Shen, Y., & Huang, L. (2020). Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions. ArXiv, abs/1909.03276.
"""

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import LogTransformLayer
from ..layers.utils import add_func


def AFN(
        linear_feature_columns, dnn_feature_columns,
        ltl_hidden_size=256,
        dnn_hidden_units=(256, 128), dnn_activation='relu',
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
        l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=0,
        seed=1024, task='binary'
):
    """Instantiates the Adaptive Factorization Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param ltl_hidden_size: integer, the number of logarithmic neurons in AFN
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer, to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(
        features,
        linear_feature_columns,
        seed=seed,
        prefix='linear',
        l2_reg=l2_reg_linear,
    )

    sparse_embedding_list, dense_value_list = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding, seed)

    # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    afn_input = tf.keras.layers.Flatten()(sparse_embedding_list)
    log_trans_layer = LogTransformLayer(ltl_hidden_size=ltl_hidden_size)(afn_input)
    afn_logit = DNN(
        dnn_hidden_units,
        dnn_activation,
        l2_reg_dnn,
        dnn_dropout,
        dnn_use_bn,
        seed=seed,
    )(log_trans_layer)

    afn_logit = tf.keras.layers.Dense(
        1,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.glorot_normal(seed)
    )(afn_logit)

    logit = add_func([linear_logit, afn_logit])
    output = PredictionLayer(task)(logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model










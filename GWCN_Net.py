# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 09:20:14 2021

@author: Maysam
"""

from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import normalized_adjacency

import numpy as np

class AGWConv(Conv):

    def __init__(
        self,
        channels,
        order=1,
        iterations=1,
        share_weights=False,
        gcn_activation="relu",
        dropout_rate=0.0,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.iterations = iterations
        self.order = order
        self.share_weights = share_weights
        self.gcn_activation = activations.get(gcn_activation)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        assert len(input_shape) >= 4
        # F = input_shape[0][-1]

        F = input_shape[0][-1]

        # Create weights for parallel stacks
        # self.kernels[k][i] refers to the k-th stack, i-th iteration
        self.kernels = []
        for k in range(self.order):
            kernel_stack = []
            current_shape = F
            for i in range(self.iterations):
                kernel_stack.append(
                    self.create_weights(
                        current_shape, F, self.channels, "GWCN_{}{}".format(k, i)
                    )
                )
                current_shape = self.channels
                if self.share_weights and i == 1:
                    # No need to continue because all weights will be shared
                    break
            self.kernels.append(kernel_stack)

        self.dropout = Dropout(self.dropout_rate, dtype=self.dtype)
        self.built = True

    def call(self, inputs):
        x, phsi, phsiInv, a_in = inputs

        output = []
        for k in range(self.order):
            output_k = x
            for i in range(self.iterations):
                output_k = self.gcs([output_k, x, phsi[k], phsiInv[k], a_in], k, i)
            output.append(output_k)
        output = K.stack(output, axis=-1)
        output = K.mean(output, axis=-1)
        output = self.activation(output)

        return output

    def create_weights(self, input_dim, input_dim_skip, channels, name):
        """
        Creates a set of weights for a GCN with skip connections.
        :param input_dim: dimension of the input space
        :param input_dim_skip: dimension of the input space for the skip connection
        :param channels: dimension of the output space
        :param name: name of the layer
        :return:
            - kernel_1, from input space of the layer to output space
            - kernel_2, from input space of the skip connection to output space
            - bias, bias vector on the output space if use_bias=True, None otherwise.
        """
        kernel_0 = self.add_weight(
            shape=(input_dim, channels),
            name=name + "_kernel_0",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        kernel_1 = self.add_weight(
            shape=(input_dim, channels),
            name=name + "_kernel_1",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        kernel_2 = self.add_weight(
            shape=(input_dim_skip, channels),
            name=name + "_kernel_2",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        bias = None
        if self.use_bias:
            bias = self.add_weight(
                shape=(channels,),
                name=name + "_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        return kernel_0, kernel_1, kernel_2, bias

    def gcs(self, inputs, stack, iteration):
        """
        Creates a graph convolutional layer with a skip connection.
        :param inputs: list of input Tensors, namely
            - input node features
            - input node features for the skip connection
            - normalized adjacency matrix;
        :param stack: int, current stack (used to retrieve kernels);
        :param iteration: int, current iteration (used to retrieve kernels);
        :return: output node features.
        """
        x, x_skip, phsi, phsiInv, a = inputs

        itr = 1 if self.share_weights and iteration >= 1 else iteration
        kernel_0, kernel_1, kernel_2, bias = self.kernels[stack][itr]
        
        # F=np.identity(2708)
        f=1
        phsi=phsi*f
        
        # output0 = K.dot(x, kernel_0)
        # output0 = ops.modal_dot(a, output0)
                
        output = K.dot(x, kernel_1)
        output = ops.modal_dot(phsiInv, output)
        # output = ops.modal_dot(F, output)
        output = ops.modal_dot(phsi, output)
        # output += output0
        
        
        skip = K.dot(x_skip, kernel_2)
        skip = self.dropout(skip)
        output += skip

        # output = K.dot(x, kernel_1)
        # output = ops.modal_dot(phsi, output)

        # skip = K.dot(x_skip, kernel_2)
        # skip = self.dropout(skip)
        # output += skip

        if self.use_bias:
            output = K.bias_add(output, bias)
        output = self.gcn_activation(output)

        return output
    

    
    
    @property
    def config(self):
        return {
            "channels": self.channels,
            "iterations": self.iterations,
            "order": self.order,
            "share_weights": self.share_weights,
            "gcn_activation": activations.serialize(self.gcn_activation),
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a, symmetric=True)




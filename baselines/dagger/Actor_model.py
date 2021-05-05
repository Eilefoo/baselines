#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import numpy as np
from baselines.a2c.utils import ortho_init


class Actor_model(keras.Model):
    def __init__(self, input_shape, nb_actions):
        self.input_shape = input_shape
        self.nb_actions = nb_actions

        self.model = self.build_model()




    def build_model(self):
        self.backbone = self.build_backbone()
        return self.build_output_layer()

    def build_backbone(self):
        robot_state_input = tf.keras.Input(shape=self.input_shape)
        # FC layers
        h1 = tf.keras.layers.Dense(units=128, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='fc1', activation='relu')(robot_state_input)
        h2 = tf.keras.layers.Dense(units=128, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='fc2', activation='relu')(h1)
        h3 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='fc3', activation='relu')(h2)
        backbone_model = tf.keras.Model(inputs=robot_state_input, outputs=[h3], name='backbone_net')
        return backbone_model

    def build_output_layer(self):
        output_layer = keras.layers.Dense(units=self.nb_actions,
                                            name='output',
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(self.backbone.outputs[0])
        model = tf.keras.Model(inputs=[self.backbone.inputs], outputs=[output_layer], name='actor_net')
        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mse'])
        return model




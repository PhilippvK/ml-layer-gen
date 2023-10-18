#
# Copyright (c) 2023 Philipp van Kempen.
#
# This file is part of ML Layer Generator.
# See https://github.com/PhilippvK/ml-layer-gen.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys

import tensorflow as tf
import numpy as np

assert len(sys.argv) >= 7

OP_PARAMS_KEYS = {
  "flatten": ["placeholder"],
  "fully_connected": ["filter_h", "filter_w", "activation"],
  "max_pool2d": ["pool_h", "pool_w", "stride_h", "stride_w", "padding", "data_format"],
  "avg_pool2d": ["pool_h", "pool_w", "stride_h", "stride_w", "padding", "data_format"],
  "conv2d": ["filters", "kernel_h", "kernel_w", "stride_h", "stride_w", "dilation_h", "dilation_w", "padding", "data_format", "activation", "groups"],
  "depthwise_conv2d": ["multiplier", "kernel_h", "kernel_w", "stride_h", "stride_w", "dilation_h", "dilation_w", "padding", "data_format", "activation"],
}


mode = sys.argv[1]
assert mode in ["tflite", "keras"]

# output_file
out_file = sys.argv[2]

# input_shape
# a,b,c,d
input_shape = tuple(map(int, sys.argv[3].split(",")))

num_layers = int(sys.argv[4])

all_op_names = []
all_op_params = []
for layer in range(num_layers):

    # op
    op_name = sys.argv[5+layer*2]
    assert op_name in OP_PARAMS_KEYS.keys(), f"Unsupported op: {op_name}"
    all_op_names.append(op_name)

    # op params
    op_params_keys = OP_PARAMS_KEYS[op_name]
    op_params_values = sys.argv[5+layer*2+1].split(",")
    assert len(op_params_keys) == len(op_params_values), f"Invalid number of params. Need {len(op_params_keys)} values. {op_params_keys}"
    op_params = dict(zip(op_params_keys, op_params_values))
    all_op_params.append(op_params)

# output_shape
# optional?
# a,b,c,d
# output_shape = tuple(map(int, sys.argv[4+num_layers*2+1].split(",")))

def get_flatten(op_name, op_params):
   return tf.keras.layers.Flatten()

# def get_fully_connected(op_name, op_params, in_shape, out_shape):
def get_fully_connected(op_name, op_params):
   assert "filter_w" in op_params
   units = int(op_params["filter_w"])
   assert "activation" in op_params
   activation = op_params["activation"]
   if len(activation) == 0:
       activation = None
   # x = tf.keras.layers.Flatten(input_shape=in_shape)
   # x = tf.keras.layers.Reshape((out_shape[0], -1))(inp)
   # return tf.keras.layers.Dense(units, activation=activation, input_shape=in_shape)
   return tf.keras.layers.Dense(units, activation=activation)

# def get_pool2d(op_name, op_params, in_shape, out_shape, type="max"):
def get_pool2d(op_name, op_params, type="max"):
   assert "pool_h" in op_params
   pool_h = int(op_params["pool_h"])
   assert "pool_w" in op_params
   pool_w = int(op_params["pool_w"])
   assert "stride_h" in op_params
   stride_h = int(op_params["stride_h"])
   assert "stride_w" in op_params
   stride_w = int(op_params["stride_w"])
   assert "padding" in op_params
   padding = op_params["padding"]
   assert "data_format" in op_params
   data_format = op_params["data_format"]  # channels_last/channels_first
   assert type in ["max", "avg"]
   op = tf.keras.layers.MaxPooling2D if type == "max" else tf.keras.layers.AveragePooling2D
   # return op(pool_size=(pool_h, pool_w), strides=(stride_h, stride_w), padding=padding, data_format=data_format, input_shape=in_shape)
   return op(pool_size=(pool_h, pool_w), strides=(stride_h, stride_w), padding=padding, data_format=data_format)

# def get_conv2d(op_name, op_params, in_shape, out_shape, depthwise=False):
def get_conv2d(op_name, op_params, depthwise=False):
   assert "kernel_h" in op_params
   kernel_h = int(op_params["kernel_h"])
   assert "kernel_w" in op_params
   kernel_w = int(op_params["kernel_w"])
   assert "stride_h" in op_params
   stride_h = int(op_params["stride_h"])
   assert "stride_w" in op_params
   stride_w = int(op_params["stride_w"])
   assert "dilation_h" in op_params
   dilation_h = int(op_params["dilation_h"])
   assert "dilation_w" in op_params
   dilation_w = int(op_params["dilation_w"])
   assert "padding" in op_params
   padding = op_params["padding"]
   assert "data_format" in op_params
   data_format = op_params["data_format"]  # channels_last/channels_first
   assert "activation" in op_params
   activation = op_params["activation"]
   if len(activation) == 0:
       activation = None
   if depthwise:
       assert "multiplier" in op_params
       multiplier = int(op_params["multiplier"])
       # print("IN", in_shape)
       # return tf.keras.layers.DepthwiseConv2D((kernel_h, kernel_w), depth_multiplier=multiplier, strides=(stride_h, stride_w), padding=padding, data_format=data_format, dilation_rate=(dilation_h, dilation_w),  activation=activation, input_shape=in_shape)
       return tf.keras.layers.DepthwiseConv2D((kernel_h, kernel_w), depth_multiplier=multiplier, strides=(stride_h, stride_w), padding=padding, data_format=data_format, dilation_rate=(dilation_h, dilation_w),  activation=activation)
   else:
       assert "filters" in op_params
       filters = int(op_params["filters"])
       assert "groups" in op_params
       groups = int(op_params["groups"]) if len(op_params["groups"]) > 0 else 1
       # return tf.keras.layers.Conv2D(filters, (kernel_h, kernel_w), strides=(stride_h, stride_w), padding=padding, data_format=data_format, dilation_rate=(dilation_h, dilation_w),  activation=activation, input_shape=in_shape)
       return tf.keras.layers.Conv2D(filters, (kernel_h, kernel_w), strides=(stride_h, stride_w), padding=padding, data_format=data_format, dilation_rate=(dilation_h, dilation_w),  activation=activation, groups=groups)

# def add_keras_layer(op_name, op_params, in_shape, out_shape):
def add_keras_layer(op_name, op_params):
    if op_name == "flatten":
        return get_flatten(op_name, op_params)
    elif op_name == "fully_connected":
        return get_fully_connected(op_name, op_params)
    elif op_name == "max_pool2d":
        return get_pool2d(op_name, op_params, type="max")
    elif op_name == "avg_pool2d":
        return get_pool2d(op_name, op_params, type="avg")
    elif op_name == "conv2d":
        return get_conv2d(op_name, op_params, depthwise=False)
    elif op_name == "depthwise_conv2d":
        return get_conv2d(op_name, op_params, depthwise=True)
    raise NotImplementedError


# Create keras_model
# inp = tf.keras.layers.Input(shape=input_shape)
inp = None
# outp = add_keras_layer(inp, op_name, op_params, in_shape=input_shape, out_shape=output_shape)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=input_shape))
for layer in range(num_layers):
    # x = add_keras_layer(all_op_names[layer], all_op_params[layer], in_shape=input_shape, out_shape=output_shape))
    model.add(add_keras_layer(all_op_names[layer], all_op_params[layer]))
# model = tf.keras.models.Model(inputs=inp, outputs=outp)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# try:
#     model.fit([], [], batch_size=500, epoch=20, validation_split=0.1)
# except:
#     print("ERR")
print(model.summary())

if mode == "keras":
    model.save(out_file)
elif mode == "tflite":
    def rep_dataset():
        for _ in range(100):
            data = np.random.rand(1, *input_shape)
            yield [data.astype(np.float32)]
        # data = [(np.array([], dtype=np.float32), 0)]
        # for a, b in data:
        #     yield [a]

    # Convert and quantize the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]

    converter.representative_dataset = rep_dataset
    # TODO: support float
    tflite_model = converter.convert()
    with open(out_file, "wb") as handle:
        handle.write(tflite_model)

    print(f"Written converted model to {out_file}.")
else:
    raise RuntimeError(f"Invalid mode: {mode}")

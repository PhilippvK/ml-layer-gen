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


assert len(sys.argv) >= 8

OP_PARAMS_KEYS = {
    "fully_connected": ["filter_h", "filter_w", "activation"],  # TODO: alias dense
    "max_pool2d": ["pool_h", "pool_w", "stride_h", "stride_w", "padding", "data_format"],
    "avg_pool2d": ["pool_h", "pool_w", "stride_h", "stride_w", "padding", "data_format"],
    "conv2d": [
        "filters",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
        "padding",
        "data_format",
        "activation",
        "groups",
    ],
    "depthwise_conv2d": [
        "multiplier",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
        "padding",
        "data_format",
        "activation",
    ],
}


mode = sys.argv[1]
assert mode in ["relay"]

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
    op_name = sys.argv[5 + layer * 2]
    assert op_name in OP_PARAMS_KEYS.keys(), f"Unsupported op: {op_name}"
    all_op_names.append(op_name)

    # op params
    op_params_keys = OP_PARAMS_KEYS[op_name]
    op_params_values = sys.argv[5 + layer * 2 + 1].split(",")
    assert len(op_params_keys) == len(
        op_params_values
    ), f"Invalid number of params. Need {len(op_params_keys)} values. {op_params_keys}"
    op_params = dict(zip(op_params_keys, op_params_values))
    all_op_params.append(op_params)

# output_shape
# optional?
# a,b,c,d
output_shape = tuple(map(int, sys.argv[4 + num_layers * 2 + 1].split(",")))

# RelayParams: {name: (shape, type)}
# RelayStatements: [str]


def gen_tensor(shape, dtype):
    shape_tuple = tuple(shape)
    return f"Tensor({shape_tuple}, {dtype})"


def get_kernel_layout(data_layout, is_depthwise=False):
    assert data_layout in ["NHWC", "NCHW"]
    if is_depthwise:
        return "HWOI" if data_layout == "NHWC" else "????"
    else:
        return "HWIO" if data_layout == "NHWC" else "????"


class RelayWriter:

    def __init__(self, input_shape, output_shape=None, quantized=True):
        self.input_shapes = [input_shape]  # TODO: allow multiple
        self.output_shapes = [output_shape]  # TODO: allow multiple
        self.quantized = quantized
        self.version = "0.0.5"
        self.stmts = []
        self.layers = []
        self.params = {}
        assert quantized, "Only quantized models are currently supported"

    def add_fully_connected_layer(self, op_params):
        raise NotImplementedError
        # assert "filter_w" in op_params
        # units = int(op_params["filter_w"])
        # assert "activation" in op_params
        # activation = op_params["activation"]
        # if len(activation) == 0:
        #     activation = None
        # # x = tf.keras.layers.Flatten(input_shape=in_shape)
        # # x = tf.keras.layers.Reshape((out_shape[0], -1))(inp)
        # # return tf.keras.layers.Dense(units, activation=activation, input_shape=in_shape)
        # return tf.keras.layers.Dense(units, activation=activation)

    def add_pool2d_layer(self, op_params, in_shape, out_shape, type="max"):
        raise NotImplementedError
        # assert "pool_h" in op_params
        # pool_h = int(op_params["pool_h"])
        # assert "pool_w" in op_params
        # pool_w = int(op_params["pool_w"])
        # assert "stride_h" in op_params
        # stride_h = int(op_params["stride_h"])
        # assert "stride_w" in op_params
        # stride_w = int(op_params["stride_w"])
        # assert "padding" in op_params
        # padding = op_params["padding"]
        # assert "data_format" in op_params
        # data_format = op_params["data_format"]  # channels_last/channels_first
        # assert type in ["max", "avg"]
        # op = tf.keras.layers.MaxPooling2D if type == "max" else tf.keras.layers.AveragePooling2D
        # # return op(pool_size=(pool_h, pool_w), strides=(stride_h, stride_w),
        # padding=padding, data_format=data_format, input_shape=in_shape)
        # return op(pool_size=(pool_h, pool_w), strides=(stride_h, stride_w), padding=padding, data_format=data_format)

    def add_conv2d_layer(self, op_params, depthwise=False):
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
        lookup_layout = {"channels_last": "NHWC", "channels_first": "NCHW"}
        # TODO: case insensitive?
        data_layout = lookup_layout[data_format]
        assert "activation" in op_params
        activation = op_params["activation"]
        if len(activation) == 0:
            activation = None
        assert activation is None, "Activations are currently not supported"
        current_idx = len(self.stmts)
        if len(self.layers) == 0:
            last_shape = self.input_shapes[0]
            last_dtype = "int8" if self.quantized else "float32"
            last_data_layout = data_layout  # TODO: this is bad?
        else:
            last_layer = self.layers[-1]
            last_shape, last_dtype, last_data_layout = last_layer
        assert last_data_layout in ["NHWC", "NCHW"]
        if last_data_layout == "NHWC":
            # _, input_h, input_w, in_channels = last_shape
            input_h, input_w, in_channels = last_shape
        elif last_data_layout == "NCHW":
            # _, in_channels, input_h, input_w = last_shape
            in_channels, input_h, input_w = last_shape
        if depthwise:
            assert "multiplier" in op_params
            multiplier = int(op_params["multiplier"])
            if multiplier != 1:
                raise NotImplementedError
            groups = in_channels
        else:
            assert "filters" in op_params
            filters = int(op_params["filters"])
            assert "groups" in op_params
            groups = int(op_params["groups"]) if len(op_params["groups"]) > 0 else 1
        assert self.quantized
        stmt = "qnn.conv2d(" if self.quantized else "nn.conv2d("
        if current_idx == 0:
            data_ref = "%input_0"
        else:
            data_ref = f"${current_idx-1}"
        kernel_idx = len(self.params)
        kernel_ref = f"%v_param_{kernel_idx+1}"
        kernel_layout = get_kernel_layout(data_layout, is_depthwise=depthwise)
        kernel_shape = "UNKNOWN"  # TODO
        # kernel_shape = get_kernel_shape(kernel_h, kernel_w, in_channels, multiplier, kernel_layout)
        kernel_dtype = "int8" if self.quantized else "float32"
        self.params[kernel_ref] = (kernel_shape, kernel_dtype)
        stmt += f"{data_ref}, {kernel_ref}, "
        if self.quantized:
            # TODO: find good values?
            input_zero_point = 0
            kernel_zero_point = 0
            input_scale = 1.0
            kernel_scale = 1.0
            stmt += f"{input_zero_point}, {kernel_zero_point}, {input_scale}f, {kernel_scale}f, "
        stmt += f"strides=[{stride_h}, {stride_w}], "
        assert padding in ["same", "valid"]
        if padding == "valid":
            pass  # implicitly
            # stmt += "padding=[0, 0, 0, 0], "  # TODO: fix
        elif padding == "same":
            dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
            dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

            def get_pad_value(a, b, c):  # TODO
                raise NotImplementedError

            pad_top, pad_bottom = get_pad_value(input_h, dilated_kernel_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, dilated_kernel_w, stride_w)
            do_pad = not (pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0)
            if do_pad:
                stmt += f"padding=[{pad_top}, {pad_left}, {pad_bottom}, {pad_right}], "
        if groups > 1:
            stmt += f"groups={groups}, "
        channels = groups if depthwise else filters
        stmt += f"channels={channels}, "
        stmt += f"kernel_size=[{kernel_h}, {kernel_w}], "
        stmt += f'data_layout="{data_layout}", '
        stmt += f'kernel_layout="{kernel_layout}", '
        out_dtype = "int32" if self.quantized else "float32"
        # TODO: downcast to int8 before next layer/at output?
        stmt += f'out_dtype="{out_dtype}"'
        stmt += ")"
        self.stmts.append(stmt)

    def add_layer(self, op_name, op_params):
        if op_name == "fully_connected":
            self.add_fully_connected_layer(op_params)
        elif op_name == "max_pool2d":
            self.add__pool2d_layer(op_params, type="max")
        elif op_name == "avg_pool2d":
            self.add_pool2d_layer(op_params, type="avg")
        elif op_name == "conv2d":
            self.add_conv2d_layer(op_params, depthwise=False)
        elif op_name == "depthwise_conv2d":
            self.add_conv2d_layer(op_params, depthwise=True)
        else:
            raise NotImplementedError

    def gen_code(self):
        ret = ""
        ret += f'#[version = "{self.version}"]\n'

        def gen_main_header():
            header = ""
            header += "def @main("
            for i, input_shape in enumerate(self.input_shapes):
                input_tensor_str = gen_tensor(input_shape, "int8" if self.quantized else "float32")
                header += f"%input_{i+1}: {input_tensor_str}, "
            for param_name in self.params:
                param_shape, param_dtype = self.params[param_name]
                param_tensor_str = gen_tensor(param_shape, param_dtype)
                header += f"{param_name}: {param_tensor_str}, "
            header += 'output_tensor_names=["Output"]'
            header += ") "
            assert len(self.output_shapes) == 1
            if self.output_shapes[0]:
                output_tensor_str = gen_tensor(self.output_shapes[0], "int8" if self.quantized else "float32")
                header += f"-> {output_tensor_str} "
            header += "{\n"
            return header

        ret += gen_main_header()

        def gen_main_body():
            body = ""
            for i, stmt in enumerate(self.stmts):
                if i == len(self.stmts) - 1:
                    body += f"    {stmt}\n"
                else:
                    body += f"    %{i} = {stmt};\n"
            return body

        ret += gen_main_body()

        def gen_main_footer():
            return "}\n"

        ret += gen_main_footer()

        return ret


if mode == "relay":
    relay_writer = RelayWriter(
        input_shape,
        # output_shape=output_shape,
        quantized=True,
    )
    for layer in range(num_layers):
        relay_writer.add_layer(all_op_names[layer], all_op_params[layer])
    code = relay_writer.gen_code()
    print("code", code)
else:
    raise RuntimeError(f"Invalid mode: {mode}")

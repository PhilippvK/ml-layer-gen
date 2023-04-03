# ml-layer-gen
Generate simple ML Models (Keras, Relay, TFLite) for testing effortlessly on the cmdline.

## Formats

- TFLite (Quantized only)

  Supported Operators: `conv2d`, `dense/fully_connected`, `max_pool2d`, `avg_pool2d`, `depthwise_conv2d`
- Keras

  Supported Operators: `conv2d`, `dense/fully_connected`, `max_pool2d`, `avg_pool2d`, `depthwise_conv2d`
- Relay (Quantized only, **WIP**)

  Supported Operators: `conv2d`, `depthwise_conv2d`, ~~`dense/fully_connected`, `max_pool2d`, `avg_pool2d`~~

## Usage

Setup python dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Syntax:

```
python3 gen_model.py FORMAT IN_SHAPE NUM_LAYERS LAYER1_OP LAYER1_ATTRS [LAYER2_OP LAYER2_ATTRS [...]]
```

## Examples

### Single Layer

#### Fully Connected

```
# Format: .tflite
# Output: out.tflite
# Input Shape: 1,1024 (1-dim)
# Number of Layers: 1
# Layer 1 Operator: fully_connected
# Layer 1 Attributes:
#   filter_height: ignore
#   filter_width: 10
#   activation = relu
python3 gen_model.py tflite out.tflite 1,1024 1 fully_connected ,10,relu
```

#### Conv2D

```
# Format: .tflite
# Output: out.tflite
# Input Shape: 32,32,3 (3-dim)
# Number of Layers: 1
# Layer 1 Operator: conv2d
# Layer 1 Attributes:
#   filters = 32
#   (kernel_h,kernel_w) = (4,4)
#   (stride_h,stride_w) = (2,2)
#   (dilation_h,dikation_w) = (1,1)
#   padding = SAME
#   format = channels_last (NHWC)
#   activation = none
#   groups = 1
python3 gen_model.py tflite out.tflite 32,32,3 1 conv2d 32,4,4,2,2,1,1,SAME,channels_last,,1
```

#### DepthwiseConv2D

```
# Format: .tflite
# Output: out.tflite
# Input Shape: 32,32,3 (3-dim)
# Number of Layers: 1
# Layer 1 Operator: depthwise_conv2d
# Layer 1 Attributes:
#   multiplier = 16
#   (kernel_h,kernel_w) = (4,4)
#   (stride_h,stride_w) = (2,2)
#   (dilation_h,dikation_w) = (1,1)
#   padding = SAME
#   format = channels_last (NHWC)
#   activation = relu
python3 gen_model.py tflite out.tflite 32,32,3 1 depthwise_conv2d 16,4,4,2,2,1,1,SAME,channels_last,relu
```

#### Pooling

```
# Format: .tflite
# Output: out.tflite
# Input Shape: 196,196,24 (3-dim)
# Number of Layers: 1
# Layer 1 Operator: max_pool/avg_pool
# Layer 1 Attributes:
#   (pool_h,pool_w) = (2,2)
#   (stride_h,stride_w) = (2,2)
#   (dilation_h,dikation_w) = (1,1)
#   padding = VALID
#   format = channels_last (NHWC)
python3 gen_model.py tflite out.tflite 196,196,24 1 max_pool2d 2,2,2,2,VALID,channels_last
python3 gen_model.py tflite out.tflite 196,196,24 1 avg_pool2d 2,2,2,2,VALID,channels_last
```

### Multiple Layers

To between conv2d/pool2d and dense layers a flatten layer might be required!

```
# Format: .tflite
# Output: out.tflite
# Input Shape: 32,32,3 (3-dim)
# Number of Layers: 4
# Layer 1 Operator: max_pool/avg_pool
# Layer 1 Attributes:
#   (pool_h,pool_w) = (2,2)
#   (stride_h,stride_w) = (2,2)
#   (dilation_h,dikation_w) = (1,1)
#   padding = VALID
#   format = channels_last (NHWC)
# Layer 2 Operator: avg_pool
# Layer 2 Attributes:
#   (pool_h,pool_w) = (2,2)
#   (stride_h,stride_w) = (2,2)
#   (dilation_h,dikation_w) = (1,1)
#   padding = VALID
#   format = channels_last (NHWC)
# Layer 3 Operator: flatten
# Layer 3 Attributes: -
# Layer 4 Operator: fully_connected
# Layer 4 Attributes:
#   filter_height: ignore
#   filter_width: 10
#   activation = relu
python3 gen_model.py tflite out2.tflite 32,32,3 4 \
    conv2d 32,4,4,2,2,1,1,SAME,channels_last,relu,1 \
    avg_pool2d 2,2,2,2,VALID,channels_last \
    flatten _ \
    fully_connected ,10,relu
```

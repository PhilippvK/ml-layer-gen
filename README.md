# ml-layer-gen
Generate simple ML Models (Keras, Relay, TFLite) for testing effortlessly on the cmdline.

## Formats

- TFLite (Quantized only)
  Supported Operators: `conv2d`, `dense/fully_connected`, `max_pool2d`, 'avg_pool2d`, `depthwise_conv2d`
- Keras
  Supported Operators: `conv2d`, `dense/fully_connected`, `max_pool2d`, 'avg_pool2d`, `depthwise_conv2d`
- Relay (Quantized only, **WIP**)
  Supported Operators: `conv2d`, `depthwise_conv2d`, ~~`dense/fully_connected`, `max_pool2d`, 'avg_pool2d`~~

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

#### Min/Max Pooling

### Multiple Layers

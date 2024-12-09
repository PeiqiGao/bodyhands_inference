# bodyhands_inference

This is the inference part for the model [BodyHands](https://github.com/cvlab-stonybrook/BodyHands)

## Installation

Follow the instruction from the [BodyHands](https://github.com/cvlab-stonybrook/BodyHands?tab=readme-ov-file#installation) to install the required libraries.

## Batch Inference

Use the following command for inference:

```
python inference.py --input /path/to/input --output /path/to/output --batch_size 32
```

## One Picture Inference

Use the following command for inference:

```
python one_inference.py --input /path/to/input --output /path/to/output
```

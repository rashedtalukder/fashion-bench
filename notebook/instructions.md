# Fashion Benchmarking Notebook

This contains a completely independent Jupyter notebook that does not depend on the source code/modules. It allows someone to learn more about OpenVINO and the different concepts for training a PyTorch model from the Fashion MNist dataset, converting to intermediate representation via OpenVINO, then optimized through quantization and hardware acceleration with benchmarks generated for viewing in the web view. This notebook is meant to be a clear and concise way to evaluate the value of OpenVINO.

## Model Training

Provide the simplest form of training the CNN for Fashion MNist with the version of Pytorch in the `/.environment.yml` file.

## Model optimization

Quantize in both int8, as well as int4 while keeping FP32 as a baseline. Automatically detect if a GPU is available. If GPU is available, make sure to test with both just the CPU kernel, and another with the CPU+GPU kernel as well.

## Benchmarking

Model benchmarks are run and stored in the respective directory following the data struction that's defined.
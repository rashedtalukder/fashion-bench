# Fashion MNist Performance Evaluator

This project uses the Fashion MNist dataset to train a computer vision model via PyTorch, uses Intel OpenVINO to generate intermediate representations (IR), and then creates quantized and optimized versions (CPU and GPU) of the models to then run benchmarks on the test datasets on each of the versions of models.

## Possible hardware targets

- Intel N97 (x86/x64 + iGPU)
- Intel XEON E3 family (x86/x64 + iGPU)
- Raspberry Pi 4B (ARMv7)
- M1 Macbook Pro (M1 Pro)

## Optimizations

There should be OpenVINO IR's available for the following:

- Intel CPU
- Intel CPU + GPU
- ARM Mac CPU
- ARMv7 CPU

## Quantization

There should be models in FP32, INT4, and INT8

## Benchmarks

- Accuracy (percent of correctly identified items in test dataset)
- Average latency between each test image
- Total time to finish running through all the data set

## Coding Style

- ALways use the conda environment defined in the `environment.yml`
- The model training, optimization, testing, and benchmarking all runs in Python.
- Always use data types
- The benchmark logs should be parsed in a HTML file with minimal CSS and Javascript. No dependencies for the HTML rendering.

## Architecture

- `/src`: The python application
- `/benchmark_logs`: JSON files of each model benchmark runs
- `/models/openvino`: The parent directory for all the OpenVINO IR files are stored for the respective quantizations.
- `/html`: Location of the HTML file that renders from the benchmark JSONs 
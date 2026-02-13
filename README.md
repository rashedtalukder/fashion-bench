# YOLO26 Clothing Detection Application

Real-time clothing detection application using Ultralytics YOLO26 on Apple Silicon.

## Features

- ✅ **Automatic Model Management**: Checks for existing trained model, trains if needed
- ✅ **Apple Silicon Acceleration**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- ✅ **Real-time Detection**: Live camera feed with bounding boxes
- ✅ **Unique Colors**: Each detected item gets a random color
- ✅ **Console Output**: Prints detected items to stdout
- ✅ **YOLO26 End-to-End**: Uses NMS-free inference for faster performance

## Detected Clothing Classes

The application detects the following clothing and accessory items:
- Person (for context)
- Backpack
- Umbrella
- Handbag
- Tie
- Suitcase

## Requirements

- Python 3.13+ (conda environment)
- macOS with Apple Silicon (M1/M2/M3)
- Webcam

## Installation

Dependencies are automatically installed:
```bash
pip install ultralytics opencv-python torch torchvision
```

## Usage

Simply run the application:
```bash
python src/app.py
```

### Application Workflow

1. **Model Check**: Looks for `models/clothing_yolo26n.pt`
2. **Training** (if needed): Trains YOLO26n for 50 epochs on clothing dataset
3. **Camera Inference**: Opens camera and starts real-time detection

### Controls

- **Q**: Quit the application
- **S**: Save current frame as screenshot

## Configuration

Edit these variables in `src/app.py`:

```python
MODEL_NAME = "yolo26n.pt"  # Base YOLO26 model
CUSTOM_MODEL_PATH = "models/clothing_yolo26n.pt"  # Trained model path
DATASET_CONFIG = "clothing.yaml"  # Dataset configuration
EPOCHS = 50  # Training epochs
IMG_SIZE = 640  # Image size for training/inference
```

## Dataset Configuration

The `clothing.yaml` file configures the training dataset. By default, it uses COCO dataset filtered for clothing classes. To use a custom dataset:

1. Organize your dataset in YOLO format:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

2. Update `clothing.yaml` with your dataset path and class names

## Output Example

```
[Frame 42] Detected 3 object(s):
  1. person (conf: 0.89) at [245, 120, 456, 678]
  2. backpack (conf: 0.76) at [310, 180, 420, 340]
  3. handbag (conf: 0.82) at [150, 450, 280, 590]
```

## YOLO26 Features Used

- **End-to-End NMS-Free**: Direct predictions without post-processing
- **MuSGD Optimizer**: Advanced training optimization
- **MPS Acceleration**: Apple Silicon GPU support
- **Dual-Head Architecture**: One-to-one head for faster inference

## Troubleshooting

### Camera Not Opening
- Check camera permissions in System Settings > Privacy & Security > Camera
- Ensure no other application is using the camera

### MPS Not Available
- Verify you're running on Apple Silicon (M1/M2/M3)
- Update PyTorch: `pip install --upgrade torch torchvision`

### Training Takes Too Long
- Reduce `EPOCHS` from 50 to 20-30
- Use smaller model: Already using `yolo26n.pt` (nano)
- Reduce `IMG_SIZE` from 640 to 480

## Model Performance

- **YOLO26n**: Fastest, lowest accuracy (~3.1M parameters)
- **YOLO26s**: Balanced (~11.1M parameters)
- **YOLO26m**: Higher accuracy (~25.9M parameters)
- **YOLO26l**: Very high accuracy (~31.7M parameters)
- **YOLO26x**: Maximum accuracy (~56.9M parameters)

Current configuration uses **YOLO26n** for optimal speed on edge devices.

## License

Uses Ultralytics YOLO26 under AGPL-3.0 license.

## References

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

# Quick Start Guide

## Running the Application

Since you don't have a custom clothing dataset yet, the application will use a pretrained COCO model for demonstration:

```bash
# Activate the environment (if not already active)
conda activate torchy

# Run the application
python src/app.py
```

## What Happens on First Run

1. **Model Check**: Looks for `models/clothing_yolo26n.pt`
2. **No custom model found**: Uses pretrained YOLO26n from COCO
3. **Camera starts**: Real-time detection begins immediately

## Expected Behavior

- Camera window opens showing live feed
- Detected clothing items highlighted with random colored boxes
- Console prints detected items with confidence scores
- Press 'Q' to quit, 'S' to save screenshot

## Training a Custom Model (Optional)

To train on your own clothing dataset:

1. **Prepare your dataset** in YOLO format:
   ```
   clothing_dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

2. **Update clothing.yaml** with your dataset path

3. **Run the app** - it will automatically train if no model exists

## Performance Notes

- **YOLO26n** is optimized for speed on Apple Silicon
- **MPS acceleration** provides GPU-level performance
- **End-to-end NMS-free** = faster inference
- Expect 15-30 FPS on MacBook Pro M1/M2/M3

## Detected Classes (COCO Pretrained)

When using the pretrained model, it can detect:
- Person
- Backpack
- Umbrella  
- Handbag
- Tie
- Suitcase

These are filtered from COCO's 80 classes to focus on clothing/accessories.

## Troubleshooting

### "Could not open camera"
```bash
# Check camera permissions
# System Settings > Privacy & Security > Camera > Terminal (enable)
```

### Import errors
```bash
# Reinstall dependencies
conda activate torchy
pip install --upgrade ultralytics opencv-python torch torchvision
```

### Slow inference
```python
# In src/app.py, lower the confidence threshold
conf=0.25  # Try 0.4 or 0.5
```

## Next Steps

1. Test with webcam to ensure camera detection works
2. Gather custom clothing dataset if needed
3. Train custom model for specific clothing types
4. Adjust confidence thresholds for your use case

Enjoy your YOLO26 clothing detector! 🎯

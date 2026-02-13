import cv2
import random
import time
import torch
from ultralytics import YOLO


# Configuration
MODEL_NAME = "yolo26m.pt"
IMG_SIZE = 640

# COCO dataset classes (80 classes)
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}


def get_random_color():
    """Generate a random BGR color for bounding boxes."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def check_mps_availability():
    """Check for GPU acceleration."""
    if torch.backends.mps.is_available():
        print("✓ Acceleration is available and will be used")
        return torch.accelerator.current_accelerator().type
    else:
        print("✗ MPS not available, using CPU")
        return "cpu"


def load_model():
    """Load pretrained YOLO26 model."""
    device = check_mps_availability()
    
    print(f"\n✓ Loading pretrained YOLO26 model: {MODEL_NAME}")
    print("This model is trained on COCO dataset with 80 object classes")
    model = YOLO(MODEL_NAME)
    
    return model, device


def run_camera_inference(model, device):
    """Run real-time inference on camera feed."""
    print("\n" + "="*60)
    print("STARTING CAMERA INFERENCE")
    print("="*60)
    print("\nPress 'q' to quit")
    print("Press 's' to save current frame")
    print("-"*60)
    
    # Initialize camera (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n✗ Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✓ Camera opened successfully")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # FPS tracking
    prev_frame_time = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✗ Error: Could not read frame")
                break
            
            # Calculate FPS
            current_frame_time = time.time()
            if prev_frame_time != 0:
                fps = 1 / (current_frame_time - prev_frame_time)
            prev_frame_time = current_frame_time
            
            # Run inference
            # YOLO26 uses end-to-end NMS-free inference by default
            results = model.predict(
                frame,
                device=device,
                conf=0.25,  # Confidence threshold
                iou=0.45,   # IOU threshold for NMS (if using one-to-many head)
                verbose=False,
                stream=False
            )
            
            # Process results
            annotated_frame = frame.copy()
            detections = []
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Get class name
                        class_name = COCO_CLASSES.get(cls, f"Class_{cls}")
                        
                        # Assign unique color per detection instance
                        # For truly random per object, use: color = get_random_color()
                        # For consistent per class: use class_colors[cls]
                        color = get_random_color()  # Unique random color per detection
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Prepare label
                        label = f"{class_name}: {conf:.2f}"
                        
                        # Draw label background
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - label_height - 10),
                            (x1 + label_width, y1),
                            color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )
                        
                        # Store detection info
                        detections.append({
                            "class": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })
            
            # Add FPS counter
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                 0.7,
                (0, 255, 0),
                2
            )
            
            # Display the frame
            cv2.imshow("YOLO26 Object Detection - Press 'q' to quit", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n\nQuitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"\n✓ Saved screenshot: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Camera released and windows closed")


def main():
    """Main application entry point."""
    print("\n" + "="*60)
    print("YOLO26 OBJECT DETECTION APPLICATION")
    print("="*60)
    print("\nThis application will:")
    print("1. Load pretrained YOLO26 model (COCO dataset)")
    print("2. Run real-time detection on camera feed")
    print("3. Detect 80 different object classes")
    print("="*60)
    
    # Load model
    model, device = load_model()
    
    # Run camera inference
    run_camera_inference(model, device)
    
    print("\n" + "="*60)
    print("APPLICATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()

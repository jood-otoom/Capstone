from ultralytics import YOLO
from pathlib import Path
import os

def view_training_summary():
    """Display training results summary."""
    
    results_dir = Path('runs/detect/accident_detection/yolov8s_quick')
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE - Accident Detection Model Ready!")
    print("=" * 70)
    
    # Model location
    model_path = results_dir / 'weights' / 'best.pt'
    print(f"\n📦 Model Location:")
    print(f"   {model_path}")
    print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Results files
    print(f"\n📊 Training Results:")
    print(f"   - Training metrics: {results_dir / 'results.csv'}")
    print(f"   - Training curves: {results_dir / 'results.png'}")
    print(f"   - Confusion matrix: {results_dir / 'confusion_matrix.png'}")
    print(f"   - Precision/Recall curves: {results_dir / 'BoxPR_curve.png'}")
    
    # Sample visualizations
    print(f"\n🖼️  Sample Visualizations:")
    print(f"   - Training batch example: {results_dir / 'train_batch0.jpg'}")
    print(f"   - Validation predictions: {results_dir / 'val_batch0_pred.jpg'}")
    
    # Load and test model
    print(f"\n🧠 Loading model...")
    model = YOLO(str(model_path))
    
    # Get model info
    print(f"\n📈 Model Information:")
    print(f"   - Model: YOLOv8 Small (yolov8s)")
    print(f"   - Task: Object Detection")
    print(f"   - Classes: 1 (Accident Detection)")
    
    return model, results_dir

def predict_on_image(model, image_path):
    """Make predictions on an image."""
    
    print(f"\n" + "=" * 70)
    print(f"🔍 MAKING PREDICTIONS")
    print("=" * 70)
    
    if not os.path.exists(image_path):
        print(f"\n⚠️  Image not found: {image_path}")
        print("Try one of these alternatives:")
        print("   1. Use any image path: predict_on_image(model, 'your_image.jpg')")
        print("   2. Predict on folder: model.predict('path/to/folder')")
        print("   3. Predict on test set: model.predict('processed_dataset_small/test/images')")
        return
    
    results = model.predict(
        source=image_path,
        conf=0.5,           # Confidence threshold
        save=True,          # Save annotated image
        project='runs/predictions',
        name='results'
    )
    
    print(f"\n✓ Predictions complete!")
    print(f"  - Detections found: {len(results[0].boxes)}")
    
    if len(results[0].boxes) > 0:
        print(f"\n  Detection Details:")
        for i, box in enumerate(results[0].boxes, 1):
            confidence = float(box.conf[0])
            print(f"    {i}. Confidence: {confidence:.2%}")
    else:
        print(f"\n  No accidents detected in this image.")
    
    print(f"\n📁 Results saved to: runs/predictions/results/")

def main():
    model, results_dir = view_training_summary()
    
    # Instructions
    print("\n" + "=" * 70)
    print("HOW TO USE THE MODEL")
    print("=" * 70)
    
    print("\n1️⃣  View Training Results:")
    print("   - Open: runs/detect/accident_detection/yolov8s_quick/results.png")
    print("   - Open: runs/detect/accident_detection/yolov8s_quick/results.csv (in Excel)")
    
    print("\n2️⃣  Predict on Images:")
    print("   Python code:")
    print("   >>> from ultralytics import YOLO")
    print("   >>> model = YOLO('runs/detect/accident_detection/yolov8s_quick/weights/best.pt')")
    print("   >>> results = model.predict(source='your_image.jpg')")
    
    print("\n3️⃣  Predict on Test Set:")
    print("   >>> results = model.predict('processed_dataset_small/test/images')")
    
    print("\n4️⃣  Command Line:")
    print("   yolo detect predict model=runs/detect/accident_detection/yolov8s_quick/weights/best.pt source=image.jpg")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
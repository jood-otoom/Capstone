from ultralytics import YOLO

def main():
    # 1. Load a pre-trained YOLOv8 model (Nano version - fastest!)
    model = YOLO('yolov8n.pt')  # Nano model for ultra-fast training

    # 2. Start training with ultra-fast settings
    results = model.train(
        data='processed_dataset_small/data.yaml',  # Use smaller dataset
        epochs=20,             # Very few epochs for speed
        imgsz=640,             # Matches our preprocessing size
        batch=32,              # Larger batch for faster processing
        device=0,              # GPU device 0 (RTX 3070 Ti)
        workers=4,             # CPU threads for data loading
        project='accident_detection', # Folder where your models will be saved
        name='yolov8n_ultra_fast', # Name of this specific training run
        patience=3             # Stops early if no improvement for 3 epochs
    )

    print("Ultra-fast training complete! Your best model is saved in: accident_detection/yolov8n_ultra_fast/weights/best.pt")

if __name__ == '__main__':
    # Required for Windows multi-processing
    main()
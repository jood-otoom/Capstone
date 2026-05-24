from ultralytics import YOLO

def main():
    # 1. Load a pre-trained YOLOv8 model
    # Options: 'yolov8n.pt' (nano - fastest), 'yolov8s.pt' (small - good balance), 'yolov8m.pt' (medium - slower)
    model = YOLO('yolov8s.pt')  # Small model for good speed/accuracy balance 

    # 2. Start training
    # Point to the data.yaml in processed_dataset folder
    results = model.train(
        data='processed_dataset_small/data.yaml',  # Use smaller dataset
        epochs=30,            # Reduced from 50 for faster training
        imgsz=640,            # Matches our preprocessing size
        batch=16,             # RTX 3070 Ti can handle 16 batch size easily
        device=0,             # GPU device 0 (RTX 3070 Ti)
        workers=4,            # CPU threads for data loading
        project='accident_detection', # Folder where your models will be saved
        name='yolov8s_quick', # Name of this specific training run
        patience=5            # Stops early if no improvement for 5 epochs (faster)
    )
    
    print("Training complete! Your best model is saved in: accident_detection/yolov8s_run1/weights/best.pt")

if __name__ == '__main__':
    # Required for Windows multi-processing
    main()

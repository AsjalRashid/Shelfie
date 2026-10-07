import os
from ultralytics import YOLO

def main():
    project_dir = "/content/drive/MyDrive/shelfie"
    data_yaml = os.path.join(project_dir, "data", "data.yaml")
    
    model = YOLO("yolov8n.pt")

    print("Starting model training...")
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=688,
        batch=16,
        name="yolov8n_custom",
        project=project_dir,
        device=0
        )

    print("Running validation...")
    metrics = model.val()
    print("Validation Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()

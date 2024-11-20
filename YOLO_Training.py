from ultralytics import YOLO

# Load a new model
model = YOLO("yolov8n.pt") 

# Use the model
if __name__ == '__main__':
    # Train the model
    results = model.train(data="yolo_config.yml", epochs=25)
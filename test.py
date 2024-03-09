from ultralytics import YOLO

# Load a model

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
if __name__ == '__main__':
    model.train(data="a.yaml", save=True)  # train the model

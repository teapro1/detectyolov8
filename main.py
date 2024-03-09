from ultralytics import YOLO

# Load a model

model = YOLO("runs/detect/train3/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
if __name__ == '__main__':
    #model.train(data="a.yaml", device='0', save=True)  # train the model
    results = model(source="ca1.mp4", save=True)  # predict on an image

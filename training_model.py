from ultralytics import YOLO

model = YOLO('../Task1_TrainingModel/runs/detect/train5/weights/best.pt')

# model.train(
#     data='/Users/wangyinghao/Library/Mobile Documents/com~apple~CloudDocs/Task1_TrainingModel/Conservatory.v1i.yolov8/data.yaml',
#     epochs=50,                                    # Set the number of training epochs
#     imgsz=640,                                    # Input image size
#     batch=16,                                     # Batch size
#     workers=4                                     # Number of data loading threads
# )

results = model('test.png', save=True)

# boxing detected conservatories
for result in results:
    result.plot()

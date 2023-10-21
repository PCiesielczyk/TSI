import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_weights.pt')

# Image
im = "C:\\Users\\piotr\\Desktop\\thesis\\Signs_test\\0000171.jpg"

# Inference
results = model(im)

print(results.pandas().xyxy[0].iloc[0])

for index, row in results.pandas().xyxy[0].iterrows():
    print(index)
    print("---")
    print(row['xmin'])

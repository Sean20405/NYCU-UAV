import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import random
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import  plot_one_box

WEIGHT = './runs/train/yolov7-lab08/weights/best.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi

cap = cv2.VideoCapture('./data/lab08_test.mp4')
if not cap.isOpened():
    raise Exception("Error: Could not open video file")

init_writer = False
while True:
    try:
        ret, image = cap.read()
        if not ret: 
            break
            
        if not init_writer:
            # Make sure output directory exists
            import os
            os.makedirs('./output', exist_ok=True)
            out = cv2.VideoWriter('./output/lab08_output.mp4', fourcc, 30.0, 
                                (image.shape[1], image.shape[0]))
            init_writer = True
            
        image_orig = image.copy()
        image = letterbox(image, (640, 640), stride=64, auto=True)[0]
        if device == "cuda":
            image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
        else:
            image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)
        with torch.no_grad():
            output = model(image)[0]
        output = non_max_suppression_kpt(output, 0.25, 0.65)[0]
        
        ## Draw label and confidence on the image
        output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
        for *xyxy, conf, cls in output:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image_orig, label=label, color=colors[int(cls)], line_thickness=1)
            
        # Write the frame into the file
        out.write(image_orig)
        
        # cv2.imshow("Detected", image_orig)
        # cv2.waitKey(1)
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        break

# Add error checking for cleanup
try:
    cap.release()
    if init_writer:
        out.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error during cleanup: {str(e)}")


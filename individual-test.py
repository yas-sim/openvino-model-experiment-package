import numpy as np
import cv2

import matplotlib.pyplot as plt

from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

import openvino_model_experiment_package as omep

# Load an IR model
model = 'intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001'
ie, net, exenet, inblobs, outblobs, inshapes, outshapes = omep.load_IR_model(model)

# Load an image and run inference
img_orig = cv2.imread('people.jpg')
res = omep.infer_ocv_image(exenet, inblobs[0], img_orig)

img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

omep.display_heatmap(res['Mconv7_stage2_L2'], overlay_img=img_orig, statistics=False)

omep.display_heatmap(res['Mconv7_stage2_L1'], statistics=False)

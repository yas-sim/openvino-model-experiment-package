# OpenVINO Model Experiment Package

## 1. Descriptioin
This project provides a set of useful functions to manipulate, analyze, display, and understand the inference result from Intel(R) Distribution of OpenVINO(TM) toolkit.  
The library can be called from an independent Python program or from an Jupyter notebook.  
The library `openvino_model_experiment_package` (`omep`) includes following functions:
 - OpenVINO simplified API : model loading, image inferencing, label reading
 - Common data processing : normalize, softmax, maxpooling, index sort, BBox NMS, heatmap NMS, Peak detection
 - Data visualize : BBox draw, statistics information, histogram, heatmap, classification reult
 - Model specific data parse : classification, SSD, YOLO, Centernet

## 2. Example of output

**Bounding box drawing**  
![bbox](./resources/bbox.png)  

**Heatmaps (DBFace)**  
![heatmap](./resources/heatmap.png)  
**Heatmap on an image**  
![heatmap-overlay](./resources/heatmap-overlay.png)  
**Heatmap with highlight and lowlight marking**  
![heatmap-highlight](./resources/heatmap-highlight.png)  
**Heatmap with peak markers**  
![heatmap-peak-marker](./resources/heatmap-peak-marker.png)  
<br>
**Heatmap from human-pose-estimation-0001 model**  
![heatmap-humanpose](./resources/heatmap-humanpose.png)  
**Histogram**  
![histogram](./resources/histogram.png)  

## 3. How to use
Place the `openvino_model_experiment_package.py` to the same directory as the Python project and import it.  
```Python
import openvino_model_experiment_package as omep
```

## 4. API
Refer to the `omep-jupyter-test.ipynb` (or `openvino_model_experiment_package.py`) to learn how to use it.  

## 5. Tested environment
- OpenVINO 2020.3 LTS
- Windows 10 1909

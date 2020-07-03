"""
OpenVINO Model Experiment Package
"""
import os
from math import exp

import numpy as np
from numpy.lib.stride_tricks import as_strided

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import maximum_filter

from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

#--------------------------------------------------------------------

def load_IR_model(model):
    """
    Load OpenVINO IR model  
    Args:
        model (string): model file path.
    Returns:
        ie (IECore object):  
        net (IENetwork object):  
        exenet (ExecutableNetwork object):  
        inblobs ([string,...]):  
        outblobs ([string,...]):  
        inshapes ([string,...]):  
        outshapes ([string,...]):  
    """
    base, ext = os.path.splitext(model)

    ie = IECore()
    net = ie.read_network(base+'.xml', base+'.bin')
    exenet = ie.load_network(net, 'CPU')
    inblobs =  (list(net.inputs.keys()))
    outblobs = (list(net.outputs.keys()))
    inshapes  = [ net.inputs [i].shape for i in inblobs  ]
    outshapes = [ net.outputs[i].shape for i in outblobs ]
    print('Input blobs: ', inblobs, inshapes)
    print('Output blobs:', outblobs, outshapes)
    return ie, net, exenet, inblobs, outblobs, inshapes, outshapes

def infer_ocv_image(exenet, image, inblob_name):
    """
    Set given image to specified input blob and run inference  
    Args:
        exenet (Executablenetwork): execet for the IR model to run
        image                     : OpenCV image (BGR, HWC)
        inblob_name (string)      : name of the input blob to set the image
    Returns:
        res: inference result
    """
    net=exenet.get_exec_graph_info()      # Obtain IENetwork
    inblob  = list(net.inputs.keys())[0]  # Obtain the name of the 1st input blob
    inshape = net.inputs[inblob].shape

    img = cv2.resize(image, (inshape[-1], inshape[-2]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1))          # HWC -> CHW, packed pixel -> planar
    res = exenet.infer(inputs={inblob_name:img})
    return res
    
def read_label_text_file(file):
    """
    Read class label text file.  
    Args:
        file (string): File name of the label
    Returns:
        label ([string]): List of labels
    """
    try:
        label = open(file).readlines()
    except OSError as e:
        label = []
    return label

#--------------------------------------------------------------------

def normalize(x):
    """
    Normalize input data.  
    Args:
        x (NDarray):
    Returns:
        Normalized NDarray
    """
    return (x-x.min())/(x.max()-x.min())    # Normalize (0.0-1.0)

def softmax(x):
    """
    Calculate softmax of the input data  
    Args:
        x (NDarray)
    Returns:
        softmax result
    """
    u = np.sum(np.exp(x))
    return np.exp(x)/u

def max_pooling(x, kernel_size, stride=1, padding=1):
    """
    Perform max pooling  
    Args:
        x (ndarray, HW)
    Returns:
        max pooling result
    """
    x = np.pad(x, padding, mode='constant')
    output_shape = ((x.shape[0] - kernel_size)//stride + 1,
                    (x.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    x_w = as_strided(x, shape=output_shape + kernel_size, strides=(stride*x.strides[0], stride*x.strides[1]) + x.strides)
    x_w = x_w.reshape(-1, *kernel_size)

    return x_w.max(axis=(1, 2)).reshape(output_shape)

def index_sort(nparray, reverse=False):
    """
    Perform sort and returns the sorted index.  
    Input data won't be modified.  
    Args:
        reverse: Sort order, True=Large->Small, False=Small->Large
    Returns:
        idx: List of index to the original array
    """
    idx = np.argsort(nparray)
    if reverse:
        return idx[::-1]
    else:
        return idx

#--------------------------------------------------------------------

def bbox_IOU(bbox1, bbox2):
    """
    Calculate IOU of 2 bboxes. bboxes are in SSD format (7 elements)  
    bbox = [id, cls, prob, x1, y1, x2, y2]  
    Args:
        bbox1 (bbox)
        bbox2 (bbox)
    Returns:
        IOU value
    """
    _xmin, _ymin, _xmax, _ymax = 3, 4, 5, 6
    width_of_overlap_area  = min(bbox1[_xmax], bbox2[_xmax]) - max(bbox1[_xmin], bbox2[_xmin])
    height_of_overlap_area = min(bbox1[_ymax], bbox2[_ymax]) - max(bbox1[_ymin], bbox2[_ymin])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    bbox1_area = (bbox1[_ymax] - bbox1[_ymin]) * (bbox1[_xmax] - bbox1[_xmin])
    bbox2_area = (bbox2[_ymax] - bbox2[_ymin]) * (bbox2[_xmax] - bbox2[_xmin])
    area_of_union = bbox1_area + bbox2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

def bbox_NMS(bboxes, threshold=0.7):
    """
    Perform non maximum suppression for bboxes to reject redundunt detections.  
    bbox = [id, cls, prob, x1, y1, x2, y2]  
    Args:
        bboxes ([bbox,...]):
        threshold (int): Threshold value of rejection
    Returns:
        NMS applied bboxes
    """
    _clsid, _prob = 1, 2
    bboxes = sorted(bboxes, key=lambda x: x[_prob], reverse=True)
    for i in range(len(bboxes)):
        if bboxes[i][_prob] == -1:
            continue
        for j in range(i + 1, len(bboxes)):
            if bbox_IOU(bboxes[i], bboxes[j]) > threshold:
                bboxes[j][_prob] = -1
    return bboxes

def hm_nms(hm, kernel_size=3):
    """
    Perform non maximum suppression for a heatmap.  
    Args:
        heat (ndarray, CHW): input heatmap
        kernel_size (int)  : kernel size
    Returns:
        NMS applied heatmap
    """
    pad = (kernel_size - 1) // 2
    hmax = np.array([max_pooling(channel, kernel_size, pad) for channel in hm])
    keep = (hmax == hm)
    return hm * keep

def detect_peaks(hm, filter_size=3, order=0.5):
    """
    Detect peaks from a heatmap.  
    Args:
        hm (ndarray, HW) : input heatmap
        filter_size (int): kernel size
        order (float) 
    Returns:
        List of detected peaks [[x0, x1, ..], [y0, y1, ..]]
    """
    local_max = maximum_filter(hm, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(hm, mask=~(hm == local_max))
    
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

def detect_peaks2(hm, threshold=1.):
    """
    Detect peaks from a heatmap.  
    Args:
        hm (ndarray, HW) : input heatmap
        threshold (float): threshold value for peak detection
    Returns:
        List of detected peaks ([x0, x1, ..], [y0, y1, ..])
    """
    hm_pool = max_pooling(hm, 3, 1, 1)
    interest_points = ((hm==hm_pool) * hm)             # screen out low-conf pixels
    flat            = interest_points.ravel()          # flatten
    indices         = np.argsort(flat)[::-1]           # index sort
    scores          = np.array([ flat[idx] for idx in indices ])
    
    scores          = scores[scores>=threshold]
    indices         = indices[:len(scores)]

    hm_height, hm_width = hm.shape
    ys = indices // hm_width
    xs = indices %  hm_width
    return (xs, ys)

#--------------------------------------------------------------------

def display_statistics(data):
    """
    Display *statistics* information of the data.  
    Args:
        data (ndarray)
    Returns:
        None
    """
    print('Max {:.2}, Min {:.2}, Mean {:.2}, Var {:.2}'.format(data.max(), data.min(), data.mean(), data.var()))

def display_histogram(data, bins=50, normalize_flg=False):
    """
    Display histogram chart. The input data will be flatten and calculate histogram.  
    Args:
        data (ndarray)
    """
    data = data.flatten()
    if normalize_flg == True:
        data = normalize(data)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=bins)
    fig.show()

def display_bboxes(objs, img, disp_label=True, label_file='voc_labels.txt'):
    """
    Display bounding boxes on the input image.  
    Args:
        objs (list of bbox) : bbox = [id, clsId, prob, x1, y1, x2, y2 ]
        img (OCV image)
        disp_label (bool)   : True=display class label (Default: True)
        label_file (string) : Label text file name
    Returns:
        None
    """
    # Read class label text file
    labels = read_label_text_file(label_file)

    img_h, img_w, _ = img.shape
    for obj in objs:
        imgid, clsid, confidence, x1, y1, x2, y2 = obj
        x1 = int( x1 * img_w )
        y1 = int( y1 * img_h )
        x2 = int( x2 * img_w )
        y2 = int( y2 * img_h )
        if confidence == -1:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), thickness=2 )
        if len(labels)>0 and disp_label==True:
            cv2.putText(img, labels[int(clsid)][:-1], (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,255,255), thickness=2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def display_heatmap(hm, overlay_img=None, col_max=4, normalize_flg=True, threshold_l=-9999, threshold_h=9999, draw_peaks=False, peak_threshold=0.7, statistics=True):
    """
    Diaplay heatmap.  
    Args:
        hm (ndarray)         : Heatmap in NCHW format
        overlay_img          : (optional) OpenCV image to display with the heatmap
        col_max (int)        : number of heatmaps in a row
        normalize_flg (bool) : True = normalize the heatmap (0.0-1.0)
        threahold_l, threshold_h (float) : Low and high threshold value to mark lowlight and highlight region
        draw_peaks (bool)    : True = Draw peak points
        peak_treshold (float): Threshold value to detect peaks
        statistics (bool)    : True = display statistics information of the heatmap data
    Returns:
        None
    """
    num_channels = hm.shape[1]

    max_grid_x = col_max
    
    if num_channels<max_grid_x:
        grid_x = num_channels
        grid_y = 1
    else:
        grid_x = max_grid_x
        grid_y = num_channels // max_grid_x + 1

    pos=1
    plt.figure(figsize=(4*grid_x, 3*grid_y))
    for ch in range(num_channels):
        _hm = hm[0, ch, :, :]                   # _hm = (H,W)
        _hm_h , _hm_w = _hm.shape
        _hm = _hm.reshape((_hm_h, _hm_w, 1))    # HWC

        if statistics == True:
            print('{} Raw : min={:.3}, max={:.3}, mean={:.3}'.format(ch, _hm.min(), _hm.max(), _hm.mean()), end='')

        # normalize
        if normalize == True:
            _hm = normalize(_hm)
            if statistics == True:
                print(', Normalized : min={:.3}, max={:.3}, mean={:.3}'.format(_hm.min(), _hm.max(), _hm.mean()), end='')

        if statistics == True:
            print()

        # Mark Highlight and lowlight
        _img = (_hm*255).astype(np.uint8)
        mask_l = np.where(_hm<threshold_l, 255, 0).astype(np.uint8)
        mask_h = np.where(_hm>threshold_h, 255, 0).astype(np.uint8)
        mask_c = cv2.bitwise_not(cv2.bitwise_or(mask_l, mask_h))
        _img   = cv2.bitwise_and(mask_c, _img)
        img_b  = cv2.bitwise_or(_img, mask_l)
        img_r  = cv2.bitwise_or(_img, mask_h)
        img_g  =                _img
        img    = cv2.merge([img_r, img_g, img_b])

        # Display the heatmap over an image (if an image is supplied)
        if not overlay_img is None:
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            overlay_img = cv2.resize(overlay_img, (img.shape[1],img.shape[0]))
            img = img//2 + overlay_img//2

        # Detect and draw peaks
        marker_color = (0,255,0)
        if draw_peaks == True:
            #peaks = detect_peaks(_hm.transpose((2,0,1)), order=peak_threshold)
            #peaks = detect_peaks(hm[:,ch,:,:], order=peak_threshold)
            peaks = detect_peaks2(_hm.reshape((_hm_h, _hm_w)), threshold=peak_threshold)
            for x, y in zip(peaks[0], peaks[1]):
                cv2.drawMarker(img, (x, y), marker_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=4, thickness=1)

        # Display the result
        plt.subplot(grid_y, grid_x, pos)
        plt.title(str(pos-1))
        plt.imshow(img)
        pos+=1
    plt.show()

def display_classification_result(res, idx, top_k, label_file='synset_words.txt'):
    """
    Display classification result.  
    Args:
        res : Image classification inference result from infer() API. This includes the probability of the classes.
        idx : Sorted index number to the probability array.
        top_k (int): Number of top results to display.
        label_file (string): Label text file name. (default: 'synset_words.txt', optional)
    Returns:
        None
    """
    # Read class label text file
    labels = read_label_text_file(label_file)

    res = res.flatten()
    for i in range(top_k):
        if len(labels)==0:
            print(i+1, idx[i]+1, res[idx[i]])
        else:
            print(i+1, idx[i]+1, res[idx[i]], labels[idx[i]][:-1])

#--------------------------------------------------------------------

def decode_classification_result(res):
    """
    Decode classification result.  
    Args:
        res: Image classification result from infer() API. This includes the probability of the classes.
    Returns:
        idx (list(int)): Sorted index number to the probability array.
    """
    res = res.flatten()
    idx = index_sort(res, reverse=True)
    return idx

def decode_ssd_result(res, threshold=0.7):
    """
    Decode SSD rsult.  
    Args:
        res: SSD inference result from infer() API. This includes the probability of the classes.
    Returns:
        objs ([bbox]): bbox = [id, clsId, prob, x1, y1, x2, y2]
    """
    res = res.reshape(res.size//7, 7)         # reshape to (x, 7)
    objs = []
    for obj in res:
        imgid, clsid, confidence, x1, y1, x2, y2 = obj
        if confidence>threshold:              # Draw a bounding box and label when confidence>threshold
            objs.append([imgid, clsid, confidence, x1, y1, x2, y2])
    return objs

def parse_yolo_region(blob, resized_image_shape, params, threshold):
    """
    Parse YOLO region. This function is intented to be called from decode_yolo_result().
    Args:
        blob               : An output blob of YOLO model inference result (one blob only).
        resized_image_shape: Shape information of the resized input image.
        params (dict)      : YOLO parameters to decode the result
        threshold (float)  : Threshold value for object rejection
    Returns:
        objs ([bbox]): bbox = [id, clsId, prob, x1, y1, x2, y2]
    """
    def entry_index(side, coord, classes, location, entry):
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

    def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
        xmin = int((x - w / 2) * w_scale)
        ymin = int((y - h / 2) * h_scale)
        xmax = int(xmin + w * w_scale)
        ymax = int(ymin + h * h_scale)
        return [class_id, confidence, xmin, ymin, xmax, ymax]

    param_num     = 3  if 'num'     not in params else int(params['num'])
    param_coords  = 4  if 'coords'  not in params else int(params['coords'])
    param_classes = 80 if 'classes' not in params else int(params['classes'])
    param_side    = int(params['side'])
    if 'anchors' not in params:
        anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 ]
    else:
        anchors = [ float(anchor) for anchor in params['anchors'].split(',') ]
    if 'mask' not in params:
        param_anchors  = anchors
        param_isYoloV3 = False
    else:
        masks          = [ int(m) for m in params['mask'].split(',')]
        param_num      = len(masks)
        param_anchors  = [ [anchors[mask*2], anchors[mask*2+1]] for mask in masks ]
        param_isYoloV3 = True

    _, _, out_blob_h, out_blob_w = blob.shape

    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = param_side * param_side

    for i in range(side_square):
        row = i // param_side
        col = i % param_side
        for n in range(param_num):
            obj_index = entry_index(param_side, param_coords, param_classes, n * side_square + i, param_coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(param_side, param_coords, param_classes, n * side_square + i, 0)

            x = (col + predictions[box_index + 0 * side_square]) / param_side
            y = (row + predictions[box_index + 1 * side_square]) / param_side
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            w = w_exp * param_anchors[n][0] / (resized_image_w if param_isYoloV3 else param_side)
            h = h_exp * param_anchors[n][1] / (resized_image_h if param_isYoloV3 else param_side)
            for j in range(param_classes):
                class_index = entry_index(param_side, param_coords, param_classes, n * side_square + i,
                                          param_coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append([0., j, confidence, x-w/2, y-h/2, x+w/2, y+h/2])
    return objects

def decode_yolo_result(res, net, inshapes, threshold):
    """
    Decode YOLO inference result.  
    Args:
        res                : YOLO inference result from infer() API. This includes the probability of the classes.
        net (IENetwork obj):
        inshapes           : Input blob shape information (list)
        threshold (float)  : Threshold value for object rejection
    Returns:
        objs ([bbox]): bbox = [id, clsId, prob, x1, y1, x2, y2]
    """
    objects=[]
    for layer_name in res:
        out_blob = res[layer_name]
        params = net.layers[layer_name].params
        params['side'] = out_blob.shape[2]
        objects += parse_yolo_region(out_blob, inshapes[0][2:], params, threshold)
    return objects

def decode_centernet(hm, dp1, dp2, threshold=0.7):
    """
    Decode Centernet inference result.  
    Args:
        hm      : Heatmap
        dp1, dp2: Displacement map
        threshold (float)  : Threshold value for object rejection
    Returns:
        objs ([bbox]): bbox = [id, clsId, prob, x1, y1, x2, y2]
    """
    hm = hm[0]
    hm_h, hm_w = hm.shape[-2:]
    hm = np.exp(hm)/(1 + np.exp(hm))
    objects=[]
    for ch in range(hm.shape[0]):
        if hm[ch].max()>0.5:
            peaks = detect_peaks2(hm[ch], threshold=threshold)
            for x, y in zip(peaks[0], peaks[1]):
                prob = hm[ch, y, x]
                if hm[ch, y, x]>0:
                    dx = dp1[0,0,y,x]
                    dy = dp1[0,1,y,x]
                    w  = dp2[0,0,y,x]
                    h  = dp2[0,1,y,x]
                    x1 = (x+dx-w/2)/hm_w
                    y1 = (y+dy-h/2)/hm_h
                    x2 = (x+dx+w/2)/hm_w
                    y2 = (y+dy+h/2)/hm_h
                    objects.append([0, ch, prob, x1, y1, x2, y2])
    return objects

print('Defined OpenVINO model experiment utility functions')
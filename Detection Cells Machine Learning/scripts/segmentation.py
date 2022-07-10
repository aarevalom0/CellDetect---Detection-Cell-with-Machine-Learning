#Libraries
from skimage.filters import threshold_otsu as otsu
import numpy as np
import cv2


def otsu_thresholding(image):
    threshold_value = otsu(image)
    binary_image = image < threshold_value
    return binary_image

def percentile_thresholding(image, percentile):
    threshold_value = np.percentile(image, percentile)
    binary_image = image < threshold_value
    return binary_image

def arbitrary_thresholding(image, threshold):
    binary_image = image < threshold
    binary_image = binary_image.astype(np.uint8) * 255
    return binary_image

def find_information(data_img,dict_info):
    id=data_img['image_id']
    path=''
    x,y,w,h=0,0,0,0
    for i in range(len(dict_info['images'])):
        if dict_info['images'][i]['id']==id:
            path='../datasets/BCCD_raw/BCCD/train/'+dict_info['images'][i]['file_name']
            break
    for i in range(len(dict_info['annotations'])):
        if dict_info['annotations'][i]['image_id']==id and dict_info['annotations'][i]['category_id']==3:
            x,y,w,h=dict_info['annotations'][i]['bbox']
        break

    return path,x,y,w,h

def sliding_window(image, window_size, minimun_trues):
    """
    Apply sliding window technique to image for detection the window with maximum number of white pixels.
    Parameters:
        image (ndarray): image
        window_size (tuple): size of the window
        minimun_trues (int): minimum number of white pixels in the window
    Returns:
        X (int): x coordinate of the window
        Y (int): y coordinate of the window
        W (int): width of the window
        H (int): height of the window
    """
    rows_image, columns_image=image.shape
    rows_window, columns_window=window_size
    W=rows_window
    H=columns_window
    maximum = 0
    for i in range(0,rows_image,5):
        for j in range(0,columns_image,5):
            if i+rows_window<=rows_image and j+columns_window<=columns_image:
                window = image[i:i+W, j:j+H]
                sum_trues = np.sum(window==0)
                if sum_trues > maximum and sum_trues > minimun_trues:
                    maximum = sum_trues
                    Y = i
                    X = j

            else:
                break
    return X,Y,W,H

def detection_white_blood_cells(image, image_id,segmentation_method, threshold=None):
    """
    Apply sliding window technique to image for detection the window with maximum number of white pixels.
    Parameters:
        image (ndarray): image
        image_id (str): id of the image
        segmentation_method (str): segmentation method
        threshold (int): threshold for the segmentation method
    Returns:
        dict_deteccion (dict): dictionary with the detection results
    """
    image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image=image[:,:,0]
    dict_deteccion={}
    dict_deteccion['image_id'] = image_id
    dict_deteccion['category_id'] = 3
    if segmentation_method == 'otsu':
        image = otsu_thresholding(image)
    elif segmentation_method == 'percentile':
        image = percentile_thresholding(image,threshold)
    elif segmentation_method == 'arbitrary':
        image = arbitrary_thresholding(image,threshold)
    bbox = sliding_window(image,(100,100),1000)
    dict_deteccion['bbox'] = bbox
    X = dict_deteccion['bbox'][0]
    Y = dict_deteccion['bbox'][1]
    W = dict_deteccion['bbox'][2]
    H = dict_deteccion['bbox'][3]
    score = np.sum(image[Y:Y+H,X:X+W])/(H*W)
    dict_deteccion['score'] = score
    return dict_deteccion
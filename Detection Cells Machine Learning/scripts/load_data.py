import glob
import os
import zipfile
import json

def extract_images(path):
  """
  Extract images from a zip file.

  Args:
    path: The path to the zip file.

  Returns:
    None
  """
  with zipfile.ZipFile(path, 'r') as zip_ref:
    extract_path = os.path.dirname(path)
    zip_ref.extractall(extract_path)



def load_data():
    """  Load the BCCD dataset.
    Returns:
        train: List of training images.
        valid: List of validation images.
        test: List of test images.
        dict_train: Dictionary containing training annotations.
        dict_val: Dictionary containing validation annotations.
        dict_test: Dictionary containing test annotations.
    """
    raw_path= "../datasets/BCCD_raw.zip"
    extract_images(raw_path)
    train = glob.glob(os.path.join('../datasets',"BCCD_raw", "BCCD","train",'*.jpg'))
    valid = glob.glob(os.path.join('../datasets',"BCCD_raw", "BCCD",'valid','*.jpg'))
    test =glob.glob(os.path.join('../datasets',"BCCD_raw", "BCCD",'test','*.jpg'))
    trainAnnotacion = open(os.path.join('../datasets',"BCCD_raw", "BCCD","train",'_annotations.coco.json'))
    dict_train = json.load(trainAnnotacion)
    valAnnotacion = open(os.path.join('../datasets',"BCCD_raw", "BCCD","valid",'_annotations.coco.json'))
    dict_val = json.load(valAnnotacion)
    testAnnotacion = open(os.path.join('../datasets',"BCCD_raw", "BCCD","test",'_annotations.coco.json'))
    dict_test = json.load(testAnnotacion)
    return train,valid,test,dict_train,dict_val,dict_test
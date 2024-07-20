import os
import glob
import shutil
from sklearn.model_selection import train_test_split

'''path to data'''
CN_PATH = 'src/data/CN'  
AD_PATH = 'src/data/AD'

TRAIN_PATH = 'src/data/train'
VALID_PATH = 'src/data/valid'

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VALID_PATH, exist_ok=True)

def split_data(source_path, dest_train, dest_valid, split_ratio=0.8):
    patients = os.listdir(source_path)
    for patient in patients:
        patient_path = os.path.join(source_path, patient)
        images = glob.glob(os.path.join(patient_path, '*.png'))
        train_images, valid_images = train_test_split(images, train_size=split_ratio)
        
        for img in train_images:
            dest = os.path.join(dest_train, patient)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(img, dest)
        
        for img in valid_images:
            dest = os.path.join(dest_valid, patient)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(img, dest)

split_data(CN_PATH, os.path.join(TRAIN_PATH, 'CN'), os.path.join(VALID_PATH, 'CN'))
split_data(AD_PATH, os.path.join(TRAIN_PATH, 'AD'), os.path.join(VALID_PATH, 'AD'))

print("Data splitting completed.")

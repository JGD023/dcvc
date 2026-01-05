import json
import os
import time
import zipfile
from multiprocessing import Pool
from shutil import copyfile

def upload_dataset(isBVI_DVC=False):
    if isBVI_DVC:
        train_path = './BVI-DVC'
        test_path = './vimeo/vimeo_test'        
    else:
        train_path = './vimeo/vimeo_train'
        test_path = './vimeo/vimeo_test'

    return train_path, test_path 

def get_pretrained_weights():
    image_model_folder = './benchmark/IntraNoAR/'
    image_models = [
        f'{image_model_folder}/ckpt_q1.pth.tar',
        f'{image_model_folder}/ckpt_q2.pth.tar',
        f'{image_model_folder}/ckpt_q3.pth.tar',
        f'{image_model_folder}/ckpt_q4.pth.tar',
        f'{image_model_folder}/ckpt_q5.pth.tar',
        f'{image_model_folder}/ckpt_q6.pth.tar',            
        f'{image_model_folder}/ckpt_q7.pth.tar',            
        f'{image_model_folder}/ckpt_q8.pth.tar',            
    ]    

    return image_models

def worker(input_command):
    print(input_command)
    os.system(input_command)

def submit_commands(commands):
    with Pool(len(commands)) as p:
        p.map(worker, commands)

import random
import shutil
import pathlib
from os import path
import pandas as pd
from types import string, double, List

def distribute_images(train_validation_split_ratio: double, csv_location: string, input_path: string):
    
    """ Distribute images that are specified in the CSV file to a train and validation folder """
    """ The folders can then be used by the TensorFlow ImageDataGenerator 
 
    :param train_validation_split_ratio: The ratio between the number of training and validation images 
    :param csv_location: The location of the CSV 
    :param input_path: The location of all the images

    """
    create_folder_structure()
    train_df = pd.read_csv(csv_location)

    for label in train_df['label'].unique():
        print(f'processing: {label}')
        pathlib.Path(f'./train/train/{label}').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'./train/validate/{label}').mkdir(parents=True, exist_ok=True)

        labels_df = train_df[train_df.label.eq(label)]
        images = labels_df['image_id'].tolist()
        random.shuffle(images)
        split_index = int(len(images) * train_validation_split_ratio)

        training_images = images[:split_index]
        validation_images = images[split_index:]

        copy_images(training_images, input_path, f'train/{label}')
        copy_images(validation_images, input_path, f'validate/{label}')


def create_folder_structure():
    """ Create the train and validation directory structure, remove the existing one"""
    if path.exists('./train'):
        shutil.rmtree('./train')
    pathlib.Path("./train/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./train/validate").mkdir(parents=True, exist_ok=True)


def copy_images(source_list: List[string], source_path: string, destination_path: string):
    """ Copy the images in the source_list from the source_path to the destination_path

    :param source_list: The list with images
    :param source_path: The current location of the images 
    :param destination_path: The location where the images should be copied

    """
    for image in source_list:
        shutil.copyfile(f'{source_path}{image}', f'./train/{destination_path}/{image}')


def copy_test_images(source_list: List[string], input_path: string):
    """ Copy the all the images in source_list from input_path to the test folder

    :param source_list: An list with all the image names
    :param input_path: The location of the images

    """
    for image in source_list:
        shutil.copyfile(f'{input_path}{image}', f'./test/1/{image}')
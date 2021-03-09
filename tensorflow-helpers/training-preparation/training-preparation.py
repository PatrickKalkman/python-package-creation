import random
import shutil
import pathlib
from os import path
import pandas as pd

def distribute_images(double train_validation_split_ratio, string csv_location, string input_path):
    
    """ Distribute images that are specified in a CSV file to a train and validate folders """
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

        copy_images(training_images, f'train/{label}', input_path)
        copy_images(validation_images, f'validate/{label}', input_path)


def create_folder_structure():
    """ """
    if path.exists('./train'):
        shutil.rmtree('./train')
    pathlib.Path("./train/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./train/validate").mkdir(parents=True, exist_ok=True)


def copy_images(source_list, destination_path, input_path):
    """

    :param source_list: 
    :param destination_path: 
    :param input_path: 

    """
    for image in source_list:
        shutil.copyfile(f'{input_path}{image}', f'./train/{destination_path}/{image}')


def copy_test_images(source_list, input_path):
    """

    :param source_list: 
    :param input_path: 

    """
    for image in source_list:
        shutil.copyfile(f'{input_path}{image}', f'./test/1/{image}')
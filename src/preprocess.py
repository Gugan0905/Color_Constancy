''' Preprocess.py
This code file is used to setup and preprocess the dataset in a suitable format for the training of our model/.

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

#import libraries
import os
import pandas as pd
from torchvision import transforms

from Custom_Dataset import Custom_Dataset

'''Class Preprocess to consolidate all the dataset setup and preprocessing. Majority involves developing the csv version of the train
and test split.'''
class Preprocess():
    def __init__(self):
        pass #empty

        ''' Function to create the entire dataset's csv mapping.
        Parameters
        x_filenames_directory - input file directory for non white illuminated images
        y_filenames_directory - input file directory for ground truth white illuminated images
        '''
    def prepare_dataset_csv(self, x_filenames_directory, y_filenames_directory):
        x_filenames = os.listdir(x_filenames_directory)
        y_filenames = os.listdir(y_filenames_directory)
        x_filenames = [x_filenames_directory + '/' + file for file in x_filenames]
        y_filenames = [y_filenames_directory + '/' + file for file in y_filenames]

        df = pd.DataFrame(columns = ['X_filenames(input_image)','Y_filenames(ground_truth)'])
        #generate the pandas dataframe

        for source_filename in y_filenames:
            for target_filename in x_filenames:
                if os.path.basename(source_filename).split('_G_AS.png')[0] in target_filename:
                    df = pd.concat([pd.DataFrame([[target_filename, source_filename]],
                                                columns = df.columns),
                                    df], ignore_index = True)

        df.to_csv('data.csv', index = False) #Saving in the current working directory

    ''' Function to split the csv into train and test sets with csv files respectively'''
    def prepare_split_csv(self, csv_path):
        df = pd.read_csv('data.csv')
        train_size = 0.8  # 80% for train
        df_randomized = df.sample(frac=1, random_state=42).reset_index(drop=True) # Randomized split
        train_dataset = df_randomized[:int(train_size * len(df_randomized))] 
        test_dataset = df_randomized[int(train_size * len(df_randomized)):]
        train_dataset.to_csv('train_data.csv', index=False)
        test_dataset.to_csv('test_data.csv', index=False)

    ''' Function to apply the initial transforms, preprocessing to the dataset and return instances of the Custom Dataet'''
    def setup_dataset(self, train_csv_file, test_csv_file):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize the image to the desired size
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #normalization based on ImageNet mean and std deviation
        ])
        train_dataset = Custom_Dataset(csv_file=train_csv_file, data_percentage = 0.1, transform=transform)
        test_dataset = Custom_Dataset(csv_file=test_csv_file, data_percentage = 0.1, transform=transform)

        return train_dataset, test_dataset


if __name__ == "__main__":
    preprocess = Preprocess()
    # Source Directories
    x_filenames_directory = 'Set1_input_images_JPG'
    y_filenames_directory = 'Set1_ground_truth_images'
    csv_path = 'data.csv'
    train_csv_file = 'train_data.csv'
    test_csv_file = 'test_data.csv'

    preprocess.prepare_dataset_csv(x_filenames_directory, y_filenames_directory)
    preprocess.prepare_split_csv(csv_path)
    print("Success!")

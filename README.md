

## Authors
- Anirudh Muthuswamy
- Gugan Kathiresan

## Operating System
- Google Colab Pro Ubuntu / V100 GPU
- macOS for file management

## Required Packages
In your command line
> cd /path/to/your/project
> pip install -r requirements.txt

## Compilation Instructions for the script files
- Download the Rendered WB dataset from the link https://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html
  - For this study we used the "Input images [a single ZIP file]" and the "Ground truth images" (with color chart pixels)
- Unzip the files and place it in the working directory
- Install all requirements from the "requirements.txt" file
- Run the "preprocess.py" file that generates all the necessary dataset csv files
- To train and peforming the test, run the  train_test.py script. 
- The train_test.py script will create new folders in your current working directory and save important information regarding your training, like:
  - Training loss graphs for the Discriminator and Generator over epochs
  - PSNR graph for Generator over epochs
  - Discriminator and Generator model .pth checkpoints for every epoch
  - Tested color corrected images over every epoch to see the progress of your training
  - Cmd Line outputs for the training loss for Discriminator and Generator
  - Cmd Line outputs for training psnr for Generator
  - Testing set psnr for generator



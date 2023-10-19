


## Details

Color constancy continues to be a topic of great attention in the field of image processing and computer vision. As deep learning techniques continue to expand and evolve, there are new methods that can be harnessed to further our understanding of Color constancy. In our project, we attempt to use conditional GANs to perform image-to-image translation as a way to achieve color constancy. With publications supporting the use of conditional GANs for image to image translation with any input to be generated as a real-world image, given that the GAN was trained on the real-world domain. We attempt to perform color constancy with an implementation of Pix-2Pix GAN trained on ground truth images of an original scene with even illumination, and given a random illumination, regenerate the image with even ground truth illumination. It is to be noted that the ground truth images are white-balanced images, and all predictions are images generated in a white-balanced illumination domain. 

## Visualization of Results:

### At 1st Epoch

<img width="885" alt="image" src="https://github.com/anirudh-muthuswamy/Color_Constancy/assets/126427302/eb66baf0-c6a6-4646-ac46-453aab24632e">

### At 10th Epoch

<img width="878" alt="image" src="https://github.com/anirudh-muthuswamy/Color_Constancy/assets/126427302/9ee1eb02-ee14-4604-ad94-44c49602a200">



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



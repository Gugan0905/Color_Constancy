''' utils.py
This code file is used to refer to common utilities that are called throughout the code files in our project.

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''
import numpy as np
import matplotlib.pyplot as plt
import os
import math

''' Class Utils that detail simple recurring utility functions'''
class Utils():
    def __init__(self):
        pass #empty

    ''' Fucntion to display an input image vs target image'''
    def imshow(self, inputs, target, figsize=(10, 5)):
        inputs = np.uint8(inputs.cpu())
        target = np.uint8(target.cpu())
        tar = np.rollaxis(target[0], 0, 3)
        inp = np.rollaxis(inputs[0], 0, 3)
        title = ['Input Image', 'Ground Truth']
        display_list = [inp, tar]
        plt.figure(figsize=figsize)

        for i in range(2):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.axis('off')
            plt.imshow(display_list[i])
        plt.axis('off') #plotting the images in a grid

        #plt.imshow(image)

    
    ''' Function to show batch size details'''
    def show_batch(self, dl):
        j=0
        for (images_a, images_b) in dl:
            j += 1
            plt.imshow(images_a, images_b) #plotting the batch images
            if j == 3:
                break

    ''' Implementing the PSNR Metric'''
    def psnr(self, outputs, label, max_val = 1):
        # Formula for psnr = 20 * log(Max pixel value) - 10 * log(MSE)

        label = label.cpu().detach().numpy() # converting to nunpy from tensor
        outputs = outputs.cpu().detach().numpy()

        diff = outputs - label
        rmse = math.sqrt(np.mean((diff)**2))
        if rmse == 0:
            return 100
        else:
            return 20 * math.log10(max_val / rmse) #Formula for psnr metric
    
    '''Function to save graphs that track the training and validaton loss and psnr'''
    def save_plot(self, train_loss, val_loss, train_psnr, val_psnr, output_dir):
        os.makedirs(output_dir, exist_ok = True)
        # Loss plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(val_loss, color='red', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{output_dir}/loss.png')
        plt.close()
        # PSNR plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_psnr, color='green', label='train PSNR dB')
        plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.savefig(f'{output_dir}/psnr.png')
        plt.close()
    

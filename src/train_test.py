''' train.py
This code file details the training script with all the parameters to ensure that 

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

import os
import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import time

from PatchGAN import PatchGAN
from utils import Utils
from preprocess import Preprocess


device = torch.device('cuda') # current GPU device

''' Class the details all the training and testing functions.
Includes functions for calculating generator and discriminator loss, psnr metrics, saving outputs,
 and evaluation of the generator for the test dataset.'''
class train_test():
        ''' Initialization function with necessary parameters'''
        def __init__(self):
            self.patchgan = PatchGAN() # PatchGAN instance
            self.preprocess = Preprocess() # Preprocess instance
            self.utils = Utils() # Util instance

            #paths to required directories
            self.train_csv_file = 'train_data.csv'
            self.test_csv_file = 'test_data.csv'
            os.makedirs('./training_weights', exist_ok = True)
            os.makedirs('./output/images', exist_ok = True)

            # Calling functions to obtain the generator, discriminator models
            self.generator, self.discriminator = self.patchgan.setup_model()
            # Calling functions to obtain the train and test dataset
            self.train_dataset, self.test_dataset = self.preprocess.setup_dataset(self.train_csv_file, self.test_csv_file)

        ''' Function to detail the loss function for generator. The generator employs adversarial loss and l1 loss 
        to evaluate the fake images generated and the reconstruction error between generated and target image 
        respectively '''

        def generator_loss(self, generated_image, target_img, G, real_target):
            adversarial_loss = nn.BCELoss() # how convincing the fake images are
            l1_loss = nn.L1Loss() # how good the reconstructon is
            gen_loss = adversarial_loss(G, real_target)
            l1_l = l1_loss(generated_image, target_img)
            gen_total_loss = gen_loss + (100 * l1_l)
            #print(gen_loss)
            return gen_total_loss

        ''' Function to detail the loss function for discriminator. The discriminator employs adversarial loss
        to evaluate whether the generated images are real enough or not'''
        def discriminator_loss(self, output, label):
            adversarial_loss = nn.BCELoss()
            disc_loss = adversarial_loss(output, label)
            return disc_loss

        ''' Function to run mean PSNR evaluation on the testing dataset'''
        def test_evaluate(self, test_loader):
            total_psnr = 0.0
            for (inputs, targets) in test_loader:
                inputs = inputs.to(device)
                generated_output = self.generator(inputs)
                total_psnr += self.utils.psnr(outputs = generated_output, label = targets)
            total_psnr = total_psnr / len(test_loader) # Mean PSNR for the entire test set

            return total_psnr

        ''' Function to start the training of the GAN on the given dataset.'''
        def start_train(self):
            
            # training hyperparameters
            learning_rate = 2e-4
            G_optimizer = torch.optim.Adam(self.generator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
            D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
            batch_size = 64 # Batch size

            # Inverse normalization to regenerate the images when they are saved
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.255]
            )

            #Train and test data loaders
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

            num_epochs = 10
            # list to track loss and psnr during training
            D_loss_plot, G_loss_plot = [], []
            G_psnr = []
            start = time.time()
            for epoch in range(1,num_epochs+1):
                print('EPOCH %d/%d' % (epoch, num_epochs))

                D_loss_list, G_loss_list = [], []

                self.discriminator.train()
                self.generator.train()
                G_running_psnr = 0.0
                for (input_img, target_img) in train_loader:

                    D_optimizer.zero_grad()
                    input_img = input_img.to(device)

                    target_img = target_img.to(device)

                    # ground truth labels real and fake
                    real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
                    fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))

                    # generator forward pass
                    generated_image = self.generator(input_img)

                    # train discriminator with fake/generated images
                    # concatenates the ground truth as well as the generated fake image
                    disc_inp_fake = torch.cat((input_img, generated_image), 1)

                    D_fake = self.discriminator(disc_inp_fake.detach())

                    D_fake_loss   =  self.discriminator_loss(D_fake, fake_target)

                    # train discriminator with real images
                    disc_inp_real = torch.cat((input_img, target_img), 1)

                    D_real = self.discriminator(disc_inp_real)
                    D_real_loss = self.discriminator_loss(D_real,  real_target)

                    # average discriminator loss
                    D_total_loss = (D_real_loss + D_fake_loss) / 2
                    D_loss_list.append(D_total_loss)
                    # compute gradients and run optimizer step
                    D_total_loss.backward()
                    D_optimizer.step()
                    print(f'\t partial train discriminator loss (single batch): %f' % (D_total_loss.data))

                    # Train generator with real labels
                    G_optimizer.zero_grad()
                    fake_gen = torch.cat((input_img, generated_image), 1)
                    G = self.discriminator(fake_gen)
                    G_loss = self.generator_loss(generated_image, target_img, G, real_target)
                    G_total_psnr = self.utils.psnr(outputs = generated_image, label = target_img)
                    G_loss_list.append(G_loss)
                    G_running_psnr += G_total_psnr
                    # compute gradients and run optimizer step
                    G_loss.backward()
                    G_optimizer.step()
                    print(f'\t partial train generator loss (single batch): %f' % (G_loss.data))

                print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f G_PSNR: %.3f' % (
                        (epoch), num_epochs, torch.mean(torch.FloatTensor(D_loss_list)),\
                        torch.mean(torch.FloatTensor(G_loss_list)), (G_running_psnr/len(train_loader))))

                D_loss_plot.append(torch.mean(torch.FloatTensor(D_loss_list)))
                G_loss_plot.append(torch.mean(torch.FloatTensor(G_loss_list)))
                G_psnr.append(G_running_psnr/len(train_loader))

                os.makedirs('./output/training_graphs', exist_ok = True)   
                output_dir =  './output/training_graphs' 

                # Loss plots.
                plt.figure(figsize=(10, 7))
                plt.plot(D_loss_plot, color='orange', label='Discriminator Train loss')
                plt.plot(G_loss_plot, color='red', label='Generator Train loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(f'{output_dir}/loss.png')
                plt.close()

                # PSNR plots.
                plt.figure(figsize=(10, 7))
                plt.plot(G_psnr, color='green', label='Generator PSNR dB')
                plt.xlabel('Epochs')
                plt.ylabel('PSNR (dB)')
                plt.legend()
                plt.savefig(f'{output_dir}/psnr.png')
                plt.close()

                #Saving the generator and discriminator model to torch checkpoints
                torch.save(self.generator.state_dict(), './training_weights/generator_epoch_%d.pth' % (epoch))
                torch.save(self.discriminator.state_dict(), './training_weights/discriminator_epoch_%d.pth' % (epoch))

                for (inputs, targets) in test_loader:
                    inputs = inputs.to(device)
                    generated_output = self.generator(inputs)
                    save_image(inv_normalize(generated_output.data[:10]), './output/images/sample_%d'%epoch + '.png', nrow=5)
                
                print("Epoch Test PSNR: ", self.test_evaluate(test_loader))
            end = time.time()
            print(f"Finished training in: {((end-start)/60):.3f} minutes")

if __name__ == "__main__":
    train_test_key = train_test()
    train_test_key.start_train()

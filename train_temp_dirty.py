"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""


import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import sys
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


import torch
from models.networks import define_G
from collections import OrderedDict
from torchvision import transforms
import itk
import datetime
import os
from tqdm import tqdm
#%%
from markdown_pdf import MarkdownPdf,Section
import pandas
def increment_experiments(opt):
    if not os.path.isfile("experiments_history.csv"):
        file = open("experiments_history.csv","w")
        args = list(vars(opt).keys())
        args.insert(1,args.pop(args.index("name")))
        args.insert(2,args.pop(args.index("model")))
        args.insert(3,args.pop(args.index("netG")))
        args.insert(4,args.pop(args.index("dataset_mode")))
        args.insert(5,args.pop(args.index("batch_size")))
        args.insert(6,args.pop(args.index("n_epochs")))
        args.insert(7,args.pop(args.index("load_size")))
        file.write("experiment date,")
        file.write(",".join(args))
        file.write("\n")
        
        vars_dict = vars(opt)
        file.write(datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
        file.write(",")
        file.write(",".join(str(vars_dict[arg]) for arg in args))
        file.close()
    else :
        df = pandas.read_csv("experiments_history.csv")     
        vars_dict = vars(opt)
        vars_dict["experiment date"] = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        df2 = pandas.DataFrame(vars_dict)
        pandas.concat([df,df2],join='outer').to_csv("experiments_history.csv")
        
        
def save_iter(res,non_contrast,contrast,epoch,save_path,is_3d=False,i=0,title=None):
    fig,ax = plt.subplots(1,3)

    ax[0].imshow(non_contrast.detach().cpu(),cmap="gray")
    ax[0].set_title(f'input')

    ax[1].imshow(contrast.detach().cpu(),cmap="gray")
    ax[1].set_title(f'expected output')

    ax[2].imshow(res.detach().cpu(),cmap="gray")
    ax[2].set_title(f'generated epoch {epoch}')
    if title is not None:
        fig.suptitle(title)
    plt.setp(ax, xticks=[], yticks=[])
    plt.savefig(save_path,bbox_inches='tight')
    plt.close()

def save_numpy_array_as_itk(array,path):
    itk_image = itk.GetImageFromArray(array)
    itk.imwrite(itk_image,path)

def volume_split(volume,split):
    for i in range(int(np.ceil(volume.shape[0] / split))):
        if i*split>volume.shape[0]:
            yield volume[i*split:]
        else :
            yield volume[i*split:(i+1)*split]
            
class SSIM_inverted:
    def __init__(self):
        self.metric = StructuralSimilarityIndexMeasure().to("cuda")
    def __call__(self,input,label):
        return 1-self.metric(input,label)

class SSIM:
    def __init__(self):
        self.metric = StructuralSimilarityIndexMeasure().to("cuda")
    def __call__(self,input,label):
        return self.metric(input,label)


class PSNR_normalized:
    def __init__(self):
        self.metric = PeakSignalNoiseRatio().to("cuda")
    def __call__(self,input,label):
        return 1-(self.metric(input,label)/50)

class PSNR:
    def __init__(self):
        self.metric = PeakSignalNoiseRatio().to("cuda")
    def __call__(self,input,label):
        return self.metric(input,label)
    
class MAE:
    def __init__(self):
        self.metric = nn.L1Loss()
    def __call__(self,input,label):
        return self.metric(input,label)

class MSE:
    def __init__(self):
        self.metric = nn.MSELoss()
    def __call__(self,input,label):
        return self.metric(input,label)

if __name__ == '__main__':
    try :    
        opt = TrainOptions().parse()   # get training options
        out_dir = Path(opt.checkpoints_dir)/opt.name
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        summary_path = out_dir/"summary"
        summary_path.mkdir(parents=True,exist_ok=True)
        

        
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.

        print('The number of training images = %d' % dataset_size)


        val_opt = deepcopy(opt)
        val_opt.phase = "val"
        # val_opt.dataset_mode="aligned"

        val_dataset = create_dataset(val_opt)
        val_dataset_size = len(val_dataset)
        
        l1 = nn.L1Loss()
        l2 = nn.MSELoss()
        ssim = SSIM_inverted()
        psnr = PSNR_normalized()
        metrics = [l1,l2,ssim,psnr]

        train_losses = []
        val_losses = []
        best_models = None
        best_epoch = None
        last_saved = 0

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        total_iters = 0                # the total number of training iterations

        with open(out_dir/"arguments.txt","w") as f:
            for arg in vars(opt):
                f.write(f"{arg} : {getattr(opt, arg)}\n")
        with open(out_dir/"command.txt","w") as f:
            f.write("python ")
            f.write(" ".join(sys.argv))   

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            model_saved = False
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            model.update_learning_rate()    # update learning rates in the beginning of every epoch.
            batch_losses = []
            model.train()
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                if opt.model.lower() == "cut" and epoch == opt.epoch_count and i == 0:
                    model.data_dependent_initialize(data)
                    model.setup(opt)               # regular setup: load and print networks; create schedulers
                
                model.set_input(data)
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                with torch.no_grad():
                    batch_losses.append([metric(model.fake_B,model.real_B).item() for metric in metrics])

                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                iter_data_time = time.time()

            train_losses.append(np.mean(batch_losses,axis=0))


            model.eval()
            batch_losses = []
            for i,data in enumerate(val_dataset):
                with torch.no_grad():
                    model.set_input(data)
                    model.forward()
                    batch_losses.append([metric(model.fake_B,model.real_B).item() for metric in metrics])

            val_losses.append(np.mean(batch_losses,axis=0))

            train_losses_arr = np.asarray(train_losses)
            val_losses_arr = np.asarray(val_losses)

            print(f"epoch : {epoch}, best epoch for each metric {np.argmin(val_losses_arr,axis=0)}, best epoch overall {np.argmin(np.mean(val_losses_arr,axis=1))}  ")
            # if epoch > (opt.n_epochs + opt.n_epochs_decay)//2:
            if epoch == np.argmin(np.mean(val_losses_arr,axis=1)):   
                best_models = model.save_networks(f"epoch_{str(epoch).zfill(len(str(opt.n_epochs+opt.n_epochs_decay)))}_avg_metric_{np.mean(val_losses_arr,axis=1)[-1]:.4f}")
                best_epoch = epoch
                model_saved = True
                last_saved=epoch
                model.save_iter(f"epoch {epoch}",summary_path/f"epoch_{epoch}_val.png")

            if epoch in np.argmin(val_losses_arr,axis=0) and not model_saved:   
                for x,y in enumerate(np.argmin(val_losses_arr,axis=0)):
                    if y==epoch:
                        model.save_networks(f"epoch_{str(epoch).zfill(len(str(opt.n_epochs+opt.n_epochs_decay)))}_{type(metrics[x]).__name__}_{val_losses_arr[y,x]:.4f}")
                last_saved=epoch
                
            if epoch-last_saved > 20 :
                model.save_networks(f"epoch_{str(epoch).zfill(len(str(opt.n_epochs+opt.n_epochs_decay)))}_periodic")
                last_saved = epoch
            
            if epoch%50 == 0 and epoch > 0:
                for i,metric in enumerate(metrics):
                    plt.close()
                    plt.plot(train_losses_arr.T[i],label=f"train metric")
                    plt.plot(val_losses_arr.T[i],label=f"val metric")
                    plt.legend()
                    plt.title(f"{type(metric).__name__} metric evolution over epochs")
                    plt.xlabel("epochs")
                    plt.ylabel(f"{type(metric).__name__}")
                    plt.savefig(out_dir/"web"/"images"/f"{type(metric).__name__}_epoch_{epoch}_losses.png")
                    plt.savefig(summary_path/f"{type(metric).__name__}_epoch_{epoch}_losses.png")
                    plt.close()
                plt.close()
                plt.plot(np.mean(train_losses_arr,axis=1),label=f"train metric")
                plt.plot(np.mean(val_losses_arr,axis=1),label=f"val metric")
                plt.legend()
                plt.title(f"All metrics average evolution over epochs")
                plt.xlabel("epochs")
                plt.ylabel(f"Metrics average")
                plt.savefig(out_dir/"web"/"images"/f"avg_epoch_{epoch}_losses.png")
                plt.savefig(summary_path/f"avg_epoch_{epoch}_losses.png")
                plt.close()

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            print(f"train l1, l2 and SSIM : {train_losses[-1]}, validation : {val_losses[-1]}")


        with open(out_dir/"commmand.txt","w") as f:
            f.write("python ")
            f.write(" ".join(sys.argv))   

    finally :
        increment_experiments(opt)
        print("training finished, saving latest model")
        if best_models is None:
            best_epoch = epoch
            best_models = model.save_networks('latest')
        else :
            model.save_networks('latest')
            
        with open(out_dir/"train_metrics.txt","w") as f:
            for i, metric in enumerate(metrics):
                f.write(f"{type(metric).__name__} : {train_losses_arr.T[i]}")
                f.write("\n")
        with open(out_dir/"val_metrics.txt","w") as f:
            for i, metric in enumerate(metrics):
                f.write(f"{type(metric).__name__} : {val_losses_arr.T[i]}")
                f.write("\n")
                
        for i,metric in enumerate(metrics):
            plt.close()
            plt.plot(train_losses_arr.T[i],label=f"train loss")
            plt.plot(val_losses_arr.T[i],label=f"val loss")
            plt.legend()
            plt.title(f"{type(metric).__name__} metric evolution over epochs")
            plt.xlabel("epochs")
            plt.ylabel(f"{type(metric).__name__}")
            plt.savefig(out_dir/"web"/"images"/f"{type(metric).__name__}_Final_losses.png")
            plt.savefig(summary_path/f"{type(metric).__name__}_Final_losses.png")
            plt.close() 
        plt.close()
        plt.plot(np.mean(train_losses_arr,axis=1),label=f"train metric")
        plt.plot(np.mean(val_losses_arr,axis=1),label=f"val metric")
        plt.legend()
        plt.title(f"All metrics average evolution over epochs")
        plt.xlabel("epochs")
        plt.ylabel(f"Metrics average")
        plt.savefig(out_dir/"web"/"images"/f"avg_Final_losses.png")
        plt.savefig(summary_path/f"avg_Final_losses.png")
        plt.close()

        # test part
        generator_name = None
        for model in best_models:
            if "net_G_A" in model:
                generator_name = model
        print(f"loading {generator_name}")
        model_dict = torch.load(generator_name)
        new_dict = OrderedDict()
        for k, v in model_dict.items():
            # load_state_dict expects keys with prefix 'module.'
            new_dict["module." + k] = v

        # make sure you pass the correct parameters to the define_G method
        generator_model = define_G(opt)#input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG, norm=opt.norm,
                                        #use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids,
                                        #n_downsampling=opt.n_downsampling, depth=opt.depth,heads=opt.heads, dropout=opt.dropout, ngf_cytran=opt.ngf_cytran)
        generator_model.load_state_dict(new_dict)
        generator_model.eval()
        generator_model.to("cuda")


        dataset_test_path = Path(opt.dataroot)
        save_volumes_dir = out_dir/"volumes"
        save_volumes_dir.mkdir(parents=True,exist_ok=True)
        
        test_metrics = [MAE(), MSE(), SSIM(), PSNR()]
        test_metrics_values = []
        trf = transforms.Compose([
            lambda x : torch.Tensor(x),
            lambda x : x.unsqueeze(1).repeat(1,opt.input_nc,1,1),
        ])

        with torch.no_grad():
            for i,patient in enumerate((dataset_test_path/"A"/"test").iterdir()):
                path = dataset_test_path/"A"/"test"
                non_contrast = itk.imread(patient, itk.SS)
                contrast = itk.imread(str(patient).replace("A","B"), itk.SS)                
                non_contrast = itk.GetArrayFromImage(non_contrast)
                contrast = itk.GetArrayFromImage(contrast)

                non_contrast = dataset.dataset.normalize(non_contrast)
                contrast = dataset.dataset.clip(contrast)
                
                non_contrast=trf(non_contrast)
                contrast=trf(contrast)
                res = []
                
                for vol in tqdm(volume_split(non_contrast,20)):
                    res.append(generator_model(vol.to("cuda")).cpu())
                res = torch.cat(res)
                
                print("non_contrast",non_contrast.shape)
                print("res",res.shape)
                
                slice = 4*(contrast.shape[0]//9)
                save_iter(res[slice,0],non_contrast[slice,0],contrast[slice,0],best_epoch,save_volumes_dir/f"test_{i}.png",title=f"Test patient example slice nb {slice}")

                non_contrast = dataset.dataset.unnormalize(contrast[:,0,:,:].squeeze().cpu().numpy())
                save_numpy_array_as_itk(non_contrast,save_volumes_dir/f"{i}_test_non_contrast.mha")
                res = dataset.dataset.unnormalize(res[:,0,:,:].squeeze().cpu().numpy())
                save_numpy_array_as_itk(res,save_volumes_dir/f"{i}_test_reconstructed.mha")
                save_numpy_array_as_itk(contrast[:,0,:,:].squeeze(),save_volumes_dir/f"{i}_test_contrast.mha")

                test_metrics_values.append([metric(res,contrast) for metric in test_metrics])
        test_metrics_avg = np.mean(test_metrics_values,axis=1)



        text = []
        text.append("# Training 2D\n")
        text.append("\n")
        text.append(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        text.append("\n")
        text.append(f"This experiment trains a {opt.model} network on each CT slice of a non contrast scan and tries to recreate the matching contrast scan.\n")
        text.append(f"This experiment uses a {opt.netG} model as generator\n")
        text.append(f"Scans Hounsfield units are clipped between {opt.min_clip} and {opt.max_clip} during preprocessing\n")
        text.append(f"This experiment uses {opt.dataset_mode} data with a batch size of {opt.batch_size} images for {opt.n_epochs+opt.n_epochs_decay} epochs with an initial lr of {opt.lr} \n")
        text.append("\n")
        text.append("TODO : Generate volume metadata properly.\n")
        text.append("\n")
        text.append("\n")
        text.append("## Data setup\n")
        text.append(f" got {len(dataset)} images in training, {len(val_dataset)} images in validation \n")
        try :
            with open(os.path.join(opt.dataroot,"split.txt"),'r') as f:
                split = f.readlines()
            text.append(f"Patient split is as follow :\n")
            text.append("".join(split).replace(",", "as").replace("\n",", "))
            text.append("\n")
        except:
            pass
        text.append("## Training metrics\n")
        text.append("\n")
        text.append('Training and validation performances are a logged with L1, L2, SSIM and PSNR metrics.\n')
        text.append("\n")
        for i,metric in enumerate(metrics):
            text.append("\n")
            text.append(f"![image]({out_dir/'web'/'images'/f'{type(metric).__name__}_Final_losses.png'})")
            text.append("\n")

        text.append("\n")
        text.append(f"![image]({out_dir/'web'/'images'/f'avg_Final_losses.png'})")
        text.append("\n")

        text.append("\n")
        text.append("\n")
        text.append("##Test results\n")
        text.append("###Metrics\n")

        text.append("| ")
        text.append(" | ".join([type(metric).__name__ for metric in metrics]))
        text.append(" |\n")

        text.append("| ")
        text.append(" | ".join(["---" for _ in range(len(metrics))]))
        text.append(" |\n")
        
        text.append("| ")
        text.append(" | ".join([np.around(metric,5) for metric in test_metrics_avg]))
        text.append(" |\n")
            
        text.append("###Qualitative results\n")
        for i in range(len(os.listdir(dataset_test_path/"test_volumesA"))):
            text.append("\n")
            text.append(f"![image]( { save_volumes_dir/f'test_{i}.png' } )")
            text.append("\n")
            text.append("\n")
        text.append("## Experiment details\n")
        for arg in vars(opt):
            text.append(f"{arg} : {getattr(opt, arg)}\n")
            text.append("\n")




        pdf = MarkdownPdf(toc_level=2, optimize=True)
        pdf.add_section(Section(''.join(text)))
        pdf.save(out_dir/f"deck_2D_{datetime.datetime.now().strftime('%m_%d_%Y_%Hh_%M')}.pdf")
            
"""
Generating images with the TTC algorithm.
"""

import os
import shutil
import sys
sys.path.append(os.path.join(os.getcwd(),'ttc_utils'))
import argparse
import pickle
import torch
from torchvision import transforms

import ttc_tools
import networks
import dataloaders
import generate_samples
import custom_transforms




def create_images(opts):
    folder = opts.folder

    # Main folder
    required_files = ['experiment_log.txt', 'step_dict.pkl']
    if not ttc_tools.check_folder(folder, required_files):
        raise OSError("Given folder does not exist or does not contain all required files.")
    images_folder = os.path.join(folder, 'images')
    new_folder = os.path.join(images_folder, opts.output_name)
    os.makedirs(new_folder)
    
    # Load step dict
    with open(os.path.join(folder, 'step_dict.pkl'), 'rb') as step_file:
        step_dict = pickle.load(step_file)
    crit_count = len(step_dict['type'])
    assert all(os.path.isfile(os.path.join(folder, 'critics', f'critic_{i+1}')) for i in range(crit_count))
    
    # Device(s)
    num_crit = len(step_dict['type'])
    num_gpus = opts.gpus if (opts.gpus > -1) else torch.cuda.device_count() # Maybe you don't want to use all the gpus available...
    if num_gpus > torch.cuda.device_count(): # Asked for more GPUs than are available.
        warning_message  = f"WARNING: {num_gpus} gpus demanded, but only {torch.cuda.device_count()} gpus available."
        warning_message += f"\nTraining will move forward using {torch.cuda.device_count()} gpus."
        print(warning_message)
        num_gpus = torch.cuda.device_count()  
    device = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    if not device: # device is empty, meaning num_gpus==0
        device.append(torch.device('cpu'))
    device = [device[(i*num_gpus)//num_crit] for i in range(num_crit)] # Now device[i] is the device for critic[i]
    
    # Data
    data_shape = step_dict['data_shape']
    loader = dataloaders.dataloader_tool(opts.source, 
                                         opts.num_samples, 
                                         shape=data_shape, 
                                         device=device[0], 
                                         train=False, 
                                         random_crop=opts.random_crop, 
                                         noise_bound=opts.noise_bound)
    load_it = iter(loader)
    batch = next(load_it)[0]
    batches = [batch.clone().cpu()]
    
    # Data corruption - for adding noise or blurring
    data_corruption = False
    if opts.noise_sigma > 0: 
        data_corruption = True
        noise_adder = custom_transforms.AddGaussianNoise(opts.noise_sigma)
        batch = noise_adder(batch)
    if opts.gaussian_blur: # i.e. opts.gaussian != ''
        data_corruption = True
        blur_params = [int(p) for p in opts.gaussian_blur.split('_')]
        ks = blur_params[0]
        if len(blur_params) == 2: #  opts.gaussian_blur = 'kernelsize_sigma'
            blurrer = transforms.GaussianBlur(kernel_size=ks, sigma=blur_params[1])
        elif len(blur_params) == 3:  #  opts.gaussian_blur = 'kernelsize_sigmamin_sigmamax'
            blurrer = transforms.GaussianBlur(kernel_size=ks, sigma=(blur_params[1], blur_params[2]))
        else:
            raise ValueError('Invalid gaussian_blur argument.')
        batch = blurrer(batch)
    if data_corruption:
        batches.append(batch.clone().cpu())
    
    # Propagator
    propagator = ttc_tools.Propagator(step_dict)
    for i in range(crit_count):
        model, dim = step_dict['critic_params'][i][0], step_dict['critic_params'][i][1]
        critic = getattr(networks, model)(dim, data_shape).to(device[i])
        critic.load_state_dict(torch.load(os.path.join(folder, 'critics', f'critic_{i+1}'), map_location=device[i]))
        critic.to(device[i])
        critic.device = device[i]
        propagator.trained_critics.append(critic)
    
    # Apply TTC to data
    ttc_tools.add_bool_item(propagator.step_dict, opts.save_steps, 'keep_samples_step')
    ttc_batches, ttc_step_indices = propagator.propagate_and_keep(batch)
    batches.extend(ttc_batches)
    
    # Save images    
    batch_names = ['orig', 'input'] if data_corruption else ['input']
    ttc_step_names = [f'step_{ind}' for ind in ttc_step_indices]
    batch_names.extend(ttc_step_names)
    generate_samples.batch_comparisons(new_folder, 
                                       batches, 
                                       individuals=opts.individuals,
                                       side_by_side=opts.side_by_side,
                                       batch_names=batch_names,
                                       grid_columns=opts.grid_columns,
                                       ext=opts.ext,
                                       grid_ext=opts.grid_ext,
                                       grid_dpi=opts.grid_dpi)
    
    # Compress
    output_name = opts.output_name
    if f'{output_name}.zip' in os.listdir(images_folder):
        i = 2
        while f'{opts.output_name}_{i}.zip' in os.listdir(images_folder):
            i += 1
        output_name = f'{opts.output_name}_{i}'
    shutil.make_archive(os.path.join(images_folder, f'{output_name}'), 'zip', new_folder)
    shutil.rmtree(new_folder)
    

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make images with TTC')
    
    # Paths
    parser.add_argument('--folder',      type=str, required=True,    help="Folder to create or from which to resume")
    parser.add_argument('--source',      type=str, required=True,    help="Either noise (for Gaussian noise source) or directory where source data is located.")
    parser.add_argument('--output_name', type=str, default='images', help="Output will be saved in zipped folder generated_images/<output_name>.zip")
    
    # What/how many images to make
    parser.add_argument('--num_samples',  type=int, default=64,    help='Number of samples for which to produce images.')
    parser.add_argument('--save_steps',   required=True,           help="Specifies after which TTC steps to create images.", choices=['all_steps', 'all_critics', 'last_step'])
    parser.add_argument('--individuals',  action='store_true',     help="If true, will create subfolder for each input sample and save all images for each sample in the corresponding subfolder.")
    parser.add_argument('--side_by_side', action='store_true',     help='Save individual pictures for each sample, side by side images showing progression through TTC steps.')
    parser.add_argument('--grid_columns', type=int, default=0,     help='If > 1, will save one picture containing all images for all samples, with grid_columns samples per row.')
    parser.add_argument('--ext',          type=str, default='jpg', help="Extension for saving individual images.")
    parser.add_argument('--grid_ext',     type=str, default='jpg', help="Extension for saving images.")
    parser.add_argument('--grid_dpi',     type=int, default=300,   help="Grid image's dpi. Increase for higher resolution.")

    # Data transforms
    parser.add_argument('--random_crop',   action='store_true',     help="If true, will reshape source data through random cropping instead of resizing.")
    parser.add_argument('--noise_sigma',   type=float, default=0.,  help="Standard deviation of noise to add to source data. Use for denoising experiments.")
    parser.add_argument('--gaussian_blur', type=str,   default='',  help="Parameters for blurring filter. Use for deblurring experiments. Specify blurring parameters as 'kernelsize_minblur_maxblur'.")
    parser.add_argument('--noise_bound',   type=float, default=-1., help="If using noise source (for generation), specifies noise bound for truncation trick. Default -1 means no truncation is used.")
    
    # Other
    parser.add_argument('--gpus', type=int, default=-1, help="Number of GPUs to use. Each critic will act on a single GPU, but different critics may be on different GPUs.")
    parser.add_argument('--seed', type=int, default=-1, help="Set random seed for reproducibility.")
    
    opts = parser.parse_args()
    create_images(opts)
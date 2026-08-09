import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from unet import Unet
from linearNoiseScheduler import LinearNoiseScheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def sample(model, scheduler, train_config, model_config, diffusion_config, task_dir):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        sample_dir = os.path.join(task_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        img.save(os.path.join(sample_dir, 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config_path
    if not os.path.isabs(config_path):
        candidate = os.path.join(script_dir, config_path)
        config_path = candidate if os.path.exists(candidate) else os.path.abspath(config_path)

    # Read the config file #
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    base_dir = os.path.dirname(config_path)
    task_dir = resolve_path(train_config['task_name'], base_dir)
    checkpoint_path = os.path.join(task_dir, train_config['ckpt_name'])
    os.makedirs(task_dir, exist_ok=True)

    # Load model with checkpoint
    model = Unet(model_config).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print('Loaded checkpoint from {}'.format(checkpoint_path))
    else:
        print('No checkpoint found at {}; using randomly initialized weights.'.format(checkpoint_path))
    model.eval()

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config, task_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
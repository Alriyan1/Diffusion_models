import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from unet import Unet
from linearNoiseScheduler import LinearNoiseScheduler

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def train(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config_path
    if not os.path.isabs(config_path):
        candidate = os.path.join(script_dir, config_path)
        config_path = candidate if os.path.exists(candidate) else os.path.abspath(config_path)

    with open(config_path,'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    base_dir = os.path.dirname(config_path)

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    dataset_path = resolve_path(dataset_config['im_path'], base_dir)
    mnist = MnistDataset('train',im_path=dataset_path)
    mnist_loader = DataLoader(mnist,batch_size=train_config['batch_size'],shuffle=True,num_workers=4)

    model = Unet(model_config).to(device)
    model.train()

    task_dir = resolve_path(train_config['task_name'], base_dir)
    os.makedirs(task_dir, exist_ok=True)
    checkpoint_path = os.path.join(task_dir, train_config['ckpt_name'])

    if os.path.exists(checkpoint_path):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(),lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)

            noise = torch.randn_like(im).to(device)

            t = torch.randint(0,diffusion_config['num_timesteps'],(im.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(im,noise,t)
            noise_pred = model(noisy_im,t)

            loss = criterion(noise_pred,noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch_idx+1,np.mean(losses),))
        torch.save(model.state_dict(), checkpoint_path)


    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm trainig')
    parser.add_argument('--config',dest='config_path',
                        default='default.yaml',type=str)

    args = parser.parse_args()
    train(args)
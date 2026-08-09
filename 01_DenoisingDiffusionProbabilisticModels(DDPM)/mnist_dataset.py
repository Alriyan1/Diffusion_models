import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, split, im_path=None, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.images = []
        self.labels = []
        self.samples = []
        self.load_images(im_path)

    def load_images(self, im_path):
        if im_path and os.path.exists(im_path):
            ims = []
            labels = []
            for d_name in tqdm(os.listdir(im_path)):
                label_dir = os.path.join(im_path, d_name)
                if not os.path.isdir(label_dir):
                    continue
                for fname in glob.glob(os.path.join(label_dir, '*.{}'.format(self.im_ext))):
                    ims.append(fname)
                    labels.append(int(d_name))
            self.images = ims
            self.labels = labels
            print('Found {} images for split {}'.format(len(ims), self.split))
            return

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
        try:
            mnist_ds = torchvision.datasets.MNIST(
                root=root_dir,
                train=(self.split == 'train'),
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
        except Exception as exc:
            raise RuntimeError(
                'images path {} does not exist and torchvision MNIST download failed: {}'.format(im_path, exc)
            ) from exc

        self.samples = [(img, int(label)) for img, label in mnist_ds]
        print('Loaded {} MNIST samples for split {}'.format(len(self.samples), self.split))

    def __len__(self):
        if self.samples:
            return len(self.samples)
        return len(self.images)

    def __getitem__(self, index):
        if self.samples:
            im, _ = self.samples[index]
            im_tensor = (2 * im) - 1
            return im_tensor

        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
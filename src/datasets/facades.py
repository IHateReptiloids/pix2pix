from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve

from PIL import Image
import torch
import torchvision

DEFAULT_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.Resize(286),
    torchvision.transforms.RandomCrop(256),
    torchvision.transforms.RandomHorizontalFlip()
])

URL = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz'


class FacadesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        root='data',
        transforms=DEFAULT_TRANSFORMS,
        device=torch.device('cpu')
    ):
        super().__init__()
        assert split in ['train', 'val', 'test']
        root = Path(root)
        if not root.exists() or not any(root.iterdir()):
            self._download(root)
        paths = sorted(root.glob(f'facades/{split}/*.jpg'))
        self.data = []
        for p in paths:
            img = Image.open(p)
            img = torchvision.transforms.functional.to_tensor(img)
            img = img.view(img.shape[0], img.shape[1], 2, img.shape[2] // 2)
            img = img.permute(2, 0, 1, 3)
            img = torch.flip(img, dims=[0])
            self.data.append(img.to(device))
        self.transforms = transforms

    def __getitem__(self, index):
        return tuple(self.transforms(self.data[index]))

    def __len__(self):
        return len(self.data)

    def _download(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        path, _ = urlretrieve(URL)
        unpack_archive(path, root, format='gztar')
        Path(path).unlink()

from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve

from PIL import Image
import torch
import torchvision

DEFAULT_TRANSFORMS = {
    'train': torchvision.transforms.ConvertImageDtype(torch.float32),
    'val': torchvision.transforms.ConvertImageDtype(torch.float32)
}

URL = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz'


class Edges2ShoesDataset(torch.utils.data.Dataset):
    def __init__(self, split, root='data', transforms=None):
        super().__init__()
        assert split in ['train', 'val']
        root = Path(root)
        if (
            not (root / 'edges2shoes').exists() or
            not any((root / 'edges2shoes').iterdir())
        ):
            self._download(root)
        paths = sorted(root.glob(f'edges2shoes/{split}/*.jpg'))
        self.data = []
        for p in paths:
            img = Image.open(p)
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = img.view(img.shape[0], img.shape[1], 2, img.shape[2] // 2)
            img = img.permute(2, 0, 1, 3)
            self.data.append(img)
        if transforms is None:
            transforms = DEFAULT_TRANSFORMS[split]
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

"""Single UVTD domain; user-approved D-11 reflect16 then existing random crop."""
from torch.utils.data import Dataset

from data.stage1_dataset import read_crop


class Stage2Dataset(Dataset):
    def __init__(self, underwater_paths):
        self.paths = list(underwater_paths)
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError('Stage2 requires a nonempty fixed manifest of unique images')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return {'Xu': read_crop(path, allow_uvtd_reflect=True), 'Xu_path': str(path)}

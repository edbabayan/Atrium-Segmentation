from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.config import CFG


class MRIProcessor:
    def __init__(self, save_root, train_ratio=0.85):
        self.save_root = Path(save_root)
        self.train_ratio = train_ratio

    @staticmethod
    def normalize(full_volume):
        mu = np.mean(full_volume)
        std = np.std(full_volume)
        normalized = (full_volume - mu) / std
        return normalized

    @staticmethod
    def standardize(normalized):
        standardized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
        return standardized

    def process(self, root_path: Path):
        all_files = list(root_path.glob('la*'))
        train_cutoff = int(len(all_files) * self.train_ratio)

        for counter, path_to_mri_data in enumerate(tqdm(all_files)):
            path_to_label_data = self.change_img_to_label_path(path_to_mri_data)
            mri = nib.load(path_to_mri_data)
            assert nib.aff2axcodes(mri.affine) == ('R', 'A', 'S')
            mri_data = mri.get_fdata()
            label_data = nib.load(path_to_label_data).get_fdata().astype(np.uint8)

            mri_data = mri_data[32:-32, 32:-32]
            label_data = label_data[32:-32, 32:-32]

            normalized = self.normalize(mri_data)
            standardized = self.standardize(normalized)

            if counter < train_cutoff:
                current_path = self.save_root / 'train' / str(counter)
            else:
                current_path = self.save_root / 'val' / str(counter)

            for i in range(standardized.shape[2]):
                slice = standardized[:, :, i]
                mask = label_data[:, :, i]
                slice_path = current_path / 'data'
                mask_path = current_path / 'masks'
                slice_path.mkdir(parents=True, exist_ok=True)
                mask_path.mkdir(parents=True, exist_ok=True)
                np.save(slice_path / f'{i}.npy', slice)
                np.save(mask_path / f'{i}.npy', mask)

    @staticmethod
    def change_img_to_label_path(path_to_mri_data):
        path_to_label_data = path_to_mri_data.as_posix().replace('imagesTr', 'labelsTr')
        path_to_label_data = Path(path_to_label_data)
        return path_to_label_data


if __name__ == "__main__":
    root_path = Path('/home/eduard/Downloads/Task02_Heart/imagesTr')
    processor = MRIProcessor(save_root=CFG.save_path)
    processor.process(root_path)

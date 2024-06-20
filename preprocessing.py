from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm

from mri_visualization import change_img_to_label_path


root_path = Path('/home/eduard/Downloads/Task02_Heart/imagesTr')


def normalize(full_volume):
    mu = np.mean(full_volume)
    std = np.std(full_volume)
    normalized = (full_volume - mu) / std
    return normalized


def standardize(normalized):
    standardized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
    return standardized


all_files = list(root_path.glob('la*'))

save_root = Path('Processed')

for counter, path_to_mri_data in enumerate(tqdm(all_files)):
    path_to_label_data = change_img_to_label_path(path_to_mri_data)
    mri = nib.load(path_to_mri_data)
    assert nib.aff2axcodes(mri.affine) == ('R', 'A', 'S')
    mri_data = mri.get_fdata()
    label_data = nib.load(path_to_label_data).get_fdata().astype(np.uint8)

    mri_data = mri_data[32:-32, 32:-32]
    label_data = label_data[32:-32, 32:-32]

    normalized = normalize(mri_data)
    standardized = standardize(normalized)

    if counter < 17:
        current_path = save_root / 'train' / str(counter)
    else:
        current_path = save_root / 'val' / str(counter)

    for i in range(standardized.shape[2]):
        slice = standardized[:, :, i]
        mask = label_data[:, :, i]
        slice_path = current_path / 'data'
        mask_path = current_path / 'label'
        slice_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)
        np.save(slice_path / f'{i}.npy', slice)
        np.save(mask_path / f'{i}.npy', mask)

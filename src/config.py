import torch
from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    save_path = root.joinpath('processed')

    training_data_dir = save_path.joinpath('train')
    validation_data_dir = save_path.joinpath('val')

    batch_size = 16
    num_workers = 4
    lr = 1e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logs_path = root.joinpath('logs')
    max_epochs = 75
    max_k = 10
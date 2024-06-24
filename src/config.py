from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    save_path = root.joinpath('processed')

    training_data_dir = save_path.joinpath('train')
    validation_data_dir = save_path.joinpath('val')

    batch_size = 16
    num_workers = 4


from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    save_path = root.joinpath('processed')

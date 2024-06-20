import numpy as np
import tkinter as tk
import nibabel as nib
from loguru import logger
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


root = Path("path_to/imagesTr")
label = Path("path_to/labelsTr")


def change_img_to_label_path(path):
    parts = list(path.parts)
    parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)


class MRIAnimationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Animation")

        self.sample_path = list(root_path.glob('la*'))[0]
        self.sample_label_path = change_img_to_label_path(self.sample_path)

        self.data = nib.load(self.sample_path)
        self.label = nib.load(self.sample_label_path)

        self.mri = self.data.get_fdata()
        self.mask = self.label.get_fdata().astype(np.uint8)

        logger.info(nib.aff2axcodes(self.data.affine))

        self.fig, self.ax = plt.subplots()

        # Create a canvas and add it to the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.mri.shape[2], interval=50, blit=False)

    def update(self, i):
        self.ax.clear()
        self.ax.imshow(self.mri[:, :, i], cmap='bone')
        mask_ = np.ma.masked_where(self.mask[:, :, i] == 0, self.mask[:, :, i])
        self.ax.imshow(mask_, alpha=0.5)


if __name__ == "__main__":
    root_path = Path(root)
    root = tk.Tk()
    app = MRIAnimationApp(root)
    root.mainloop()

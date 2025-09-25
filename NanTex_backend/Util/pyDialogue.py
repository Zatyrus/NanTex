import tkinter as tk
from tkinter import filedialog
import os


class pyDialogue:
    def __init__(self):
        pass

    @staticmethod
    def askDIR(query_title: str = "Please select a directory"):
        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        DIR_path = filedialog.askdirectory(title=query_title)
        return DIR_path

    @staticmethod
    def askDIRS(query_title: str = "Please select a directory to scan"):
        directory = pyDialogue.askDIR(query_title=query_title)
        return [f.path for f in os.scandir(directory) if f.is_dir()]

    @staticmethod
    def askFILE(query_title: str = "Please select a file"):
        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        FILE_path = filedialog.askopenfilename(title=query_title)
        return FILE_path

    @staticmethod
    def askFILES(query_title: str = "Please select a number of files"):
        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        FILE_path = filedialog.askopenfilenames(title=query_title)
        return FILE_path

    @staticmethod
    def askSAVEASFILE():
        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        FILE_path = filedialog.asksaveasfilename(
            defaultextension="*",
            title="Save as...",
            confirmoverwrite=True,
            filetypes=[
                ("Text files", "*.txt"),
                ("Json files", "*.json"),
                ("CSV files", "*.csv"),
                ("Npy files", "*.npy"),
                ("Pickled files", "*.pkl"),
                ("All files", "*"),
            ],
        )
        return FILE_path

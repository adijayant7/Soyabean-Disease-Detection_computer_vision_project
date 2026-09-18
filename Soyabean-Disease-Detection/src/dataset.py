from pathlib import Path
import csv

from PIL import Image
from torch.utils.data import Dataset    

class SoybeanDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = Path(csv_file)
        self.transform = transform

        self.image_paths = []
        self.labels = []
        with open(
                    self.csv_file,
                    "r",
                    newline="",
                    encoding="utf-8"
                ) as file:
        
                    reader = csv.DictReader(file)
        
                    for row in reader:
                        self.image_paths.append(row["image_path"])
                        self.labels.append(int(row["label"]))

    def __len__(self):
            return len(self.image_paths)
    def __getitem__(self, index):

            image_path = self.image_paths[index]
            label = self.labels[index]

            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label
        


 
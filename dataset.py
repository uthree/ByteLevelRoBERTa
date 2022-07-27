import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_pathes):
        self.data = []
        for path in file_pathes:
            with open(path) as f:
                self.data += f.read().split("\n")
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

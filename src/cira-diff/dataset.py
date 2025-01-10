from torch.utils.data import Dataset
import zarr
import torch 

class ZarrDataset(Dataset):
    """
    This is a new zarr instance of the dataset that loads all data into CPU memory to minimize I/O overhead.
    """
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        
        # Load data into CPU memory
        self.input_images = torch.tensor(self.data['input_images'][:], dtype=torch.float16, device='cpu')
        self.output_images = torch.tensor(self.data['output_images'][:], dtype=torch.float16, device='cpu')
        self.length = self.input_images.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Access data from memory (still on the CPU)
        return self.output_images[idx], self.input_images[idx]
from torch.utils.data import Dataset
import zarr
import torch 

class ZarrDataset(Dataset):
    """This is a new zarr instance of the dataset chunked at the desired batchsize to ensure the GPU is fed. """
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        self.length = self.data['input_images'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load data lazily
        input_image = self.data['input_images'][idx]
        output_image = self.data['output_images'][idx]
        return torch.tensor(output_image, dtype=torch.float16),torch.tensor(input_image, dtype=torch.float16)
from dataloader_ import diffusion_dataloader
from monai.data import Dataset, PersistentDataset
from monai import transforms
import pandas as pd
from torch.utils.data import DataLoader
import tqdm as tqdm
from monai.data.image_reader import ITKReader
if __name__ == "__main__":
    json_path = r"C:\bjorn\train_2.json"
    data = pd.read_json(json_path)
    data = data.to_dict(orient='records')
    c_dir = r"C:\bjorn\cache_dir"
    INPUT_SHAPE_AE = (128, 128, 128)
    itk_reader = ITKReader()
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'file_name'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image'], reader=itk_reader),
        transforms.EnsureChannelFirstD(keys=['image']),
    ])
    trainset = PersistentDataset(data=data, transform=transforms_fn, cache_dir=c_dir)
    train_loader = DataLoader(dataset=trainset, 
                                num_workers=1, 
                                batch_size=8, 
                                shuffle=True, 
                                persistent_workers=True, 
                                pin_memory=True)
    for i, batch in enumerate(train_loader):
        if i%100 == 0:
            print(batch)

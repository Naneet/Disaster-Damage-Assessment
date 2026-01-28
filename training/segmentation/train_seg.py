import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation

from loss_utils import dice_loss, loss_fn
from utils import get_pre_disaster_image_mask_pairs
from segmentation_dataset import XBD_Building_Segmentation_Dataset
from train_test_step import train_step, test_step



tier1 = get_pre_disaster_image_mask_pairs('/kaggle/input/xbd-dataset/xbd/tier1')
tier3 = get_pre_disaster_image_mask_pairs('/kaggle/input/xbd-dataset/xbd/tier3')
tier1.extend(tier3)
train_images = tier1

test_images = get_pre_disaster_image_mask_pairs('/kaggle/input/xbd-dataset/xbd/test')


train_transform = T.Compose([
    T.Resize((640, 640)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(size=(512, 512)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),  
])

val_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToImage(), 
    T.ToDtype(torch.float32, scale=True),
])

train_dataset = XBD_Building_Segmentation_Dataset(train_images, train_transform)
test_dataset = XBD_Building_Segmentation_Dataset(test_images, val_transform)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=2,
                              prefetch_factor=3,
                              pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=8,
                              shuffle=False,
                              num_workers=2,
                              prefetch_factor=3,
                              pin_memory=True)

                        
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512',
    num_labels=1,
    ignore_mismatched_sizes=True
).to('cuda')

model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
bce_loss = torch.nn.BCEWithLogitsLoss()

current = 0
num_epochs = 15

start = 1 + current
epochs = num_epochs + current + 1

print(f"Running from {start} to {epochs-1}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("Ready to train!!")

for epoch in range(start, epochs):
    train_step(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch=epoch,
        dataloader=train_dataloader
    )

    test_step(
        model=model,
        loss_fn=loss_fn,
        epoch=epoch,
        dataloader=test_dataloader,
    )
    checkpoint = model.state_dict()
    torch.save(obj=checkpoint, f=f"Building_SegFormer_{epoch}_epochs.pth")
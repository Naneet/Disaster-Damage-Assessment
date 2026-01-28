import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

from models.classifier import BuildingClassificationModel
from train_test_step import train_step, test_step



torch.backends.cudnn.deterministic = True

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((96,96)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((96,96))
])


train_dataset = ImageFolder(
    root="/kaggle/input/xbd-buildings/building_dataset/train",
    transform=train_transform
)

test_dataset = ImageFolder(
    root="/kaggle/input/xbd-buildings/building_dataset/test",
    transform=test_transform
)


num_workers = 2
batch_size = 16
prefetch_factor = 16

train_dataloader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              prefetch_factor=prefetch_factor,
                              num_workers=num_workers)

test_dataloader = DataLoader(dataset=test_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              prefetch_factor=prefetch_factor,
                              num_workers=num_workers)


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = "cuda"

model = DeepFakeModel().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(params=model.parameters(),
                            lr=1e-4,
                            weight_decay=1e-3)

num_epochs = 20
steps_per_epoch = len(train_dataloader)
total_steps = num_epochs * steps_per_epoch


current = 0

start = 1 + current
epochs = num_epochs + current + 1

print(f"Running from {start} to {epochs-1}")

print("Ready to train!!")

for epoch in range(start, epochs):
    train_loss, train_acc = train_step(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch=epoch,
        dataloader=train_dataloader
    )

    test_loss, test_acc = test_step(
        model=model,
        loss_fn=loss_fn,
        epoch=epoch,
        dataloader=test_dataloader,
    )
    print(f'Epoch: {epoch} | Train Loss:{train_loss:.4f} | Train acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%')

    if epoch%1==0:
        checkpoint = {
            "model_state_dict" : model.state_dict(),
            "train_acc" : train_acc,
            "test_acc" : test_acc,
            "train_loss" : train_loss,
            "test_loss" : test_loss,
            "epoch" : epoch,
            "optimizer" : optimizer.state_dict(),
            "scheduler" : scheduler.state_dict()
        }
        torch.save(obj=checkpoint, f=f"XBD_Building_{epoch}_epochs_{test_acc:.2f}.pth")
        del checkpoint

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print(classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes
))
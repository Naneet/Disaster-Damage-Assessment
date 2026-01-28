from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

def train_step(model, optimizer, dataloader, loss_fn, epoch, device="cuda"):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for img, label in tqdm(dataloader):
        img, label = img.to(device), label.to(device)

        with autocast('cuda'):
            y_logits = model(img).squeeze(-1)

            loss = loss_fn(y_logits, label)
            batch_loss = loss.item()
            train_loss += batch_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        preds = y_logits.argmax(1)
        correct += (preds == label).sum().item()
        total += label.numel()

        del img, label, loss, y_logits

    avg_loss = train_loss / len(dataloader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc


def test_step(model, dataloader, loss_fn, epoch, device="cuda"):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)


            with autocast('cuda'):
                y_logits = model(img).squeeze(-1)

                loss = loss_fn(y_logits, label)
                batch_loss = loss.item()
                test_loss += batch_loss

            preds = y_logits.argmax(1)
            correct += (preds == label).sum().item()
            total += label.numel()

            del img, label, loss, y_logits

    avg_loss = test_loss / len(dataloader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc
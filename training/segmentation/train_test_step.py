from torch.amp import autocast, GradScaler
import torch
from tqdm import tqdm

device = 'cuda'

scaler = GradScaler('cuda')

def train_step(model, dataloader, optimizer, loss_fn, epoch, device='cuda'):
    model.train()
    total_iou, train_loss = 0, 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        with autocast('cuda'):
            outputs = model(img)
            logits = outputs.logits
            
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode="nearest"
            ).long()
            
            loss = loss_fn(logits, mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        
        preds = (torch.sigmoid(logits) > 0.5).int()
        intersection = ((preds == 1) & (mask == 1)).sum()
        union = ((preds == 1) | (mask == 1)).sum()

        batch_iou = intersection.float() / (union.float() + 1e-6)
        total_iou += batch_iou.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{batch_iou.item():.4f}"
        )

    iou = total_iou/len(dataloader)
    avg_loss = train_loss/len(dataloader)
    print(f"Epoch: {epoch} | Train Loss: {avg_loss:.4f} | IoU: {iou:.4f}")

def test_step(model, dataloader, loss_fn, epoch, device='cuda'):
    model.eval()
    total_iou, test_loss = 0, 0

    pbar = tqdm(dataloader, desc=f"Test Epoch {epoch}")

    with torch.inference_mode():
        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda'):
                outputs = model(img)
                logits = outputs.logits
                
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1).float(),
                    size=logits.shape[-2:],
                    mode="nearest"
                ).long()
                
                loss = loss_fn(logits, mask)
    
            test_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).int()
            intersection = ((preds == 1) & (mask == 1)).sum()
            union = ((preds == 1) | (mask == 1)).sum()

            batch_iou = intersection.float() / (union.float() + 1e-6)
            total_iou += batch_iou.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{batch_iou.item():.4f}"
            )

    iou = total_iou/len(dataloader)
    avg_loss = test_loss/len(dataloader)
    print(f"Epoch: {epoch} | Test Loss: {avg_loss:.4f} | IoU: {iou:.4f}")
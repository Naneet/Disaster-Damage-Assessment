from transformers import SegformerForSemanticSegmentation
from models.classifier import BuildingClassificationModel
from torchvision import transforms
import torch


def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    building_seg_model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b0-finetuned-ade-512-512',
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    checkpoint = torch.load('./models/Building_SegFormer_14_epochs.pth')
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint.items()
    }

    building_seg_model.load_state_dict(state_dict)
    del checkpoint, state_dict

    seg_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    classifier_model = BuildingClassificationModel().to(device)
    checkpoint = torch.load('./models/XBD_Building_20_epochs_83.30_96x.pth')["model_state_dict"]
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint.items()
    }

    classifier_model.load_state_dict(state_dict)
    del checkpoint, state_dict

    classifier_transform = transforms.Compose([
        transforms.Resize((128,128)),
    ])

    return building_seg_model, classifier_model, seg_transform, classifier_transform
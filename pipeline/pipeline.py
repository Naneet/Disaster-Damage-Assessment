import torch
import numpy as np
import cv2
from torchvision.transforms import functional as F



class DamageAssessmentPipeline:
    def __init__(
        self, 
        building_seg_model,
        seg_transform,
        classifier_model,
        classifier_transform,
        device
    ):
        self.building_seg_model = building_seg_model.eval()
        self.seg_transform = seg_transform

        self.classifier_model = classifier_model.eval()
        self.classifier_transform = classifier_transform

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.COLORS = {
            "no_damage": (0, 255, 0),       # green
            "minor_damage": (255, 255, 0),  # yellow
            "major_damage": (255, 165, 0),  # orange
            "destroyed": (255, 0, 0)        # red
        }
        self.LABELS = {
            0: 'destroyed',
            1: 'major_damage',
            2: 'minor_damage',
            3: 'no_damage'
        }

    def run(self, pre_image, post_image):

        building_mask = self._segment_building(pre_image)
        buildings, coords, labels = self._extract_buildings(post_image, building_mask)
        results, logits = self._classify_building(buildings)
        detections = self._build_detections(results, logits, coords)
        overlay_image = self._render_mask_overlay(post_image, labels, results)
        summary = self._aggregate(results)

        return {
            'overlay_image': overlay_image,
            'raw_mask': building_mask,
            'detections': detections,
            'summary': summary
        }

    def _segment_building(self, image):
        h,w,c = np.array(image).shape
        image = self.seg_transform(image)
        with torch.inference_mode():
            mask = self.building_seg_model(image.unsqueeze(0).to(self.device)).logits.cpu().squeeze(0)
        
        mask = F.resize(img=mask, size=(h,w))
        print(mask.shape, h,w,c)
        return mask

    def _extract_buildings(self, image, mask):
        image = F.to_tensor(image)
        mask = mask.permute(1,2,0).numpy()
        
        binary_mask = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
        buildings = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            crop_img = image[:, y:y+h, x:x+w]
            buildings.append(self.classifier_transform(crop_img))

        buildings = torch.stack(buildings)
        return buildings, stats, labels

    def _classify_building(self, buildings):
        with torch.inference_mode():
            logits =  self.classifier_model(buildings.to(self.device)).squeeze(-1).cpu()
        results = logits.argmax(1)
        return results, logits

    def _build_detections(self, results, logits, coords):
        detections = []
        probs = logits.softmax(1)
        for i in range(1, len(coords)):
            x, y, w, h, area = coords[i]
            cls = self.LABELS[results[i-1].item()]
            conf = probs[i-1][results[i-1].item()]
            detections.append({'bbox': (x,y,w,h), 'class': cls, 'confidence': conf})
        return detections
    
    def _render_mask_overlay(self, image, labels, results, alpha=0.4):
        image = np.array(image)
        overlay = image.copy()
        for i, cls_id in enumerate(results, start=1):
            cls = self.LABELS[int(cls_id)]
            color = self.COLORS[cls]
    
            ys, xs = np.where(labels == i)  # it is returning a np ndarray but it needs to be indices which can be given to overlay for changing the colour so look into it :)
            overlay[ys, xs] = color
    
        blended = cv2.addWeighted(
            overlay, alpha,
            image, 1 - alpha,
            0
        )
    
        return blended

    def _aggregate(self, results):
        unique_elements, counts = torch.unique(results, return_counts=True)
        return {
            self.LABELS[u.item()]: c.item()
            for u, c in zip(unique_elements, counts)
        }
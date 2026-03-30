"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset
import torchvision
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import numpy as np

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, root: str = "./data", split: str = "trainval", transform=None, download: bool = True):
        self.root = root
        self.split = split
        self.transform = transform
        
        # We leverage torchvision for downloading/caching images and trimaps.
        self.tv_dataset_imgs = torchvision.datasets.OxfordIIITPet(
            root=root, split=split, target_types="category", download=download
        )
        self.tv_dataset_masks = torchvision.datasets.OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=False
        )
        
        self.xml_dir = os.path.join(root, "oxford-iiit-pet", "annotations", "xmls")
        if not hasattr(self.tv_dataset_imgs, '_images'):
            # Handling older/newer torchvision versions where private variable differs
            self.image_files = [] 
        else:
            self.image_files = [os.path.basename(p) for p in self.tv_dataset_imgs._images]
        
    def __len__(self):
        return len(self.tv_dataset_imgs)
        
    def __getitem__(self, idx):
        img, class_idx = self.tv_dataset_imgs[idx]
        _, trimap = self.tv_dataset_masks[idx]
        
        # Determine XML path
        if hasattr(self, 'image_files') and len(self.image_files) > idx:
            img_name = self.image_files[idx]
            xml_name = img_name.rsplit('.', 1)[0] + '.xml'
        else:
            # Fallback for dynamic fetching
            import glob
            xml_name = "unknown.xml"
            pass # Simplified for assignment skeleton

        xml_path = os.path.join(self.xml_dir, xml_name)
        
        w_img, h_img = img.size
        
        # Default bounding box if XML is missing (some images lack XML in Oxford Pet)
        bbox = torch.tensor([112.0, 112.0, 224.0, 224.0], dtype=torch.float32)
        
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bndbox = root.find('.//bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Normalize to [0, 1] relative to width/height
                w = (xmax - xmin) / w_img * 224.0
                h = (ymax - ymin) / h_img * 224.0
                cx = ((xmin / w_img) * 224.0) + (w / 2.0)
                cy = ((ymin / h_img) * 224.0) + (h / 2.0)
                bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        
        trimap = torch.tensor(np.array(trimap), dtype=torch.long)
        # trimap has pixels 1 (foreground), 2 (background), 3 (not classified).
        # We remap to 0, 1, 2 for CE loss
        trimap = trimap - 1
        
        import torchvision.transforms as T
        if not self.transform:
            base_tr = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            img_tensor = base_tr(img)
            # Nearest neighbor for trimap to avoid interpolating labels
            trimap_tr = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)(trimap.unsqueeze(0).float()).squeeze(0).long()
        else:
            # Assumes albumentations format handled internally
            img_tensor = self.transform(img)
            trimap_tr = trimap 
            
        return {
            "image": img_tensor,
            "class_label": torch.tensor(class_idx, dtype=torch.long),
            "bbox_target": bbox,
            "segmentation_mask": trimap_tr
        }
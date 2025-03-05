import os
import random
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def split_dataset(seed, base_path, original_path, train_ratio=0.7, val_ratio=0.15, add_originals=False):
    random.seed(seed)
    
    image_groups = {}
    base_path_imgs = sorted(os.listdir(os.path.join(base_path, "images")))
    for file in base_path_imgs:
        if file.endswith(('.jpg', '.png', '.jpeg',  'bmp', 'tiff', 'gif', 'webp')):
            name_parts = file.split(".")[0].split("_")
            base_name = "_".join(name_parts[:-1])

            if base_name not in image_groups:
                image_groups[base_name] = []
            image_groups[base_name].append(file)
    
    all_keys = list(image_groups.keys())
    random.shuffle(all_keys)
    
    train_cutoff = int(len(all_keys) * train_ratio)
    val_cutoff = train_cutoff + int(len(all_keys) * val_ratio)
    
    train_set = set(all_keys[:train_cutoff])
    val_set = set(all_keys[train_cutoff:val_cutoff])
    test_set = set(all_keys[val_cutoff:])
    
    if add_originals:
        for file in os.listdir(original_path):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                base_name = os.path.splitext(file)[0]
                if base_name in train_set:
                    image_groups[base_name].append(file)
                elif base_name in val_set:
                    image_groups[base_name].append(file)
                elif base_name in test_set:
                    image_groups[base_name].append(file)
    
    return train_set, val_set, test_set, image_groups

class CustomDataset(Dataset):
    def __init__(self, image_set, image_groups, base_path, transform=None):
        self.image_list = []
        for key in image_set:
            self.image_list.extend(image_groups[key])
        self.transform = transform
        self.image_dir = os.path.join(base_path, "images")
        self.label_dir = os.path.join(base_path, "classes")
        self.bbox_dir = os.path.join(base_path, "bboxes")
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace(image_name.split('.')[-1], 'txt'))
        bbox_path = os.path.join(self.bbox_dir, image_name.replace(image_name.split('.')[-1], 'txt'))
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        with open(label_path, 'r') as f:
            label = f.readline().strip()
        
        with open(bbox_path, 'r') as f:
            bbox = [float(line.strip()) for line in f]
        
        return image, label, bbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_path", type=str, required=True, help="Path to augmented images")
    parser.add_argument("--original_path", type=str, required=True, help="Path to original images")
    parser.add_argument("--add_originals", action="store_true", default=False, help="Include original images in dataset split")
    args = parser.parse_args()
    
    train_set, val_set, test_set, image_groups = split_dataset(args.seed, args.base_path, args.original_path, add_originals=args.add_originals)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_dataset = CustomDataset(train_set, image_groups, args.base_path, transform=transform)
    val_dataset = CustomDataset(val_set, image_groups, args.base_path, transform=transform)
    test_dataset = CustomDataset(test_set, image_groups, args.base_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    # test
    for i, (image, label, bbox) in enumerate(train_loader):
        print(image.shape, label, bbox)
        if i == 0:
            break
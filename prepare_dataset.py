import os
import numpy as np
import argparse
from PIL import Image, ImageDraw 
import json
import cv2
import re


class PrepareDataset:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.output_annotation_path = args.output_annotation

        self.annotations = []
        self.coco_annotations = []
        self.llm_annotations = []
        self.images = []
        self.categories = {}
        self.category_id = 1
        self.annotation_id = 1

        self.image_id_map = {}
        self.image_id_counter = 1
        self.image_groups = {}

        self.init_dirs()

    def init_dirs(self):
        self.images_dir = os.path.join(self.data_dir, "images")
        self.bboxes_dir = os.path.join(self.data_dir, "bboxes")
        self.classes_dir = os.path.join(self.data_dir, "classes")

    def parse_dataset(self):
        pattern = re.compile(r"(.+?_\d+)_(\d+)\..+")

        images_paths = sorted(os.listdir(self.images_dir))
        for filename in images_paths:
            match = pattern.match(filename)
            if match:
                image = Image.open(os.path.join(self.images_dir, filename))
                base_name, index = match.groups()
                index = int(index)
                img_extention = os.path.splitext(filename)[1]
                
                if base_name not in self.image_groups or index > self.image_groups[base_name]:
                    self.image_groups[base_name] = index

                bbox_file = os.path.join(self.bboxes_dir, filename.replace(img_extention, ".txt"))
                class_file = os.path.join(self.classes_dir, filename.replace(img_extention, ".txt"))
                
                with open(bbox_file, "r") as f_bbox, open(class_file, "r") as f_class:
                    bbox = [float(f_bbox.readline().strip()) for _ in range(4)]
                    class_name = f_class.readline().strip()
                
                if base_name not in self.image_id_map or index > self.image_groups[base_name]:
                    image_name = f"{base_name}"
                    self.image_id_map[base_name] = self.image_id_counter
                    self.images.append({
                        "id": self.image_id_counter,
                        "file_name": image_name,
                        "width": image.width, 
                        "height": image.height
                    })
                    self.image_id_counter += 1
                
                image_id = self.image_id_map[base_name]

                if class_name not in self.categories:
                    self.categories[class_name] = self.category_id
                    self.category_id += 1
                
                self.annotations.append({
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": self.categories[class_name],
                    "bbox": bbox,
                    "area": (bbox[2]-bbox[0])*(bbox[3]-bbox[1]),
                    "iscrowd": 0
                })
                self.annotation_id += 1

        categories_list = [{"id": cid, "name": cname} for cname, cid in self.categories.items()]
        self.images = [f"{img['file_name']}_{self.image_groups[img['file_name']]}{img_extention}" for img in self.images]

        self.coco_annotations["images"] = self.images
        self.coco_annotations["annotations"] = self.annotations
        self.coco_annotations["categories"] = categories_list

    def count_objects(self):
        for filename in images_paths:
            # match = pattern.match(filename)
            # if match:
                # base_name, index = match.groups()   

            base_name = filename.split(".")[0]
            img_extention = os.path.splitext(filename)[1]

            bbox_file = os.path.join(self.bboxes_dir, filename.replace(img_extention, ".txt"))
            class_file = os.path.join(self.classes_dir, filename.replace(img_extention, ".txt"))
            
            with open(bbox_file, "r") as f_bbox, open(class_file, "r") as f_class:
                bbox = [float(f_bbox.readline().strip()) for _ in range(4)]
                class_name = f_class.readline().strip()

            prompt = f"<image>\nсколько на картинке изображено {class_name}?"
            answer = 1
            
            self.llm_annotations.append({
                "id": base_name,
                "image": filename,
                "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": answer
                }]
            })
    
    def prepare_dataset(self):
        self.parse_dataset()
        

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset. Should contain images, bboxes and classes folders")
    parser.add_argument("--output_annotation", type=str, default="data", help="Annotation name to save the prepared dataset in COCO format")
    args = parser.parse_args()

    preparation = PrepareDataset(args)
    preparation.prepare_dataset()

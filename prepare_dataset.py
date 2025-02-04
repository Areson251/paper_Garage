import os
import numpy as np
import argparse
from PIL import Image, ImageDraw 
import json
import cv2
import re
from pycocotools.coco import COCO


class PrepareDataset:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.output_annotation_path = args.output_annotation

        self.annotations = []
        self.coco_annotations = {}
        self.llm_annotations = []
        self.images = []
        self.categories = {}
        self.category_id = 1
        self.annotation_id = 1

        self.image_id_map = {}
        self.image_id_counter = 1
        self.image_groups = {}
        self.group_annotations = {}  # Хранение аннотаций для каждой группы

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
                
                bbox_file = os.path.join(self.bboxes_dir, filename.replace(img_extention, ".txt"))
                class_file = os.path.join(self.classes_dir, filename.replace(img_extention, ".txt"))
                
                with open(bbox_file, "r") as f_bbox, open(class_file, "r") as f_class:
                    bbox = [float(f_bbox.readline().strip()) for _ in range(4)]
                    class_name = f_class.readline().strip()
                
                self.images.append({
                    "id": self.image_id_counter,
                    "file_name": filename,
                    "width": image.width, 
                    "height": image.height
                })
                image_id = self.image_id_counter
                self.image_id_counter += 1
                
                if class_name not in self.categories:
                    self.categories[class_name] = self.category_id
                    self.category_id += 1
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": self.categories[class_name],
                    "bbox": bbox,
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "iscrowd": 0
                }
                self.annotations.append(annotation)
                
                if base_name not in self.group_annotations:
                    self.group_annotations[base_name] = []
                
                # Добавляем все предыдущие аннотации этой группы
                for ann in self.group_annotations[base_name]:
                    self.annotation_id += 1
                    ann2add = ann.copy()
                    ann2add["id"] = self.annotation_id 
                    ann2add["image_id"] = image_id
                    self.annotations.append(ann2add)
                    
                self.group_annotations[base_name].append(annotation)
                self.annotation_id += 1

        categories_list = [{"id": cid, "name": cname} for cname, cid in self.categories.items()]
        # self.images = [f"{img['file_name']}_{self.image_groups[img['file_name']]}{img_extention}" for img in self.images]

        self.coco_annotations["images"] = self.images
        self.coco_annotations["annotations"] = self.annotations
        self.coco_annotations["categories"] = categories_list

        with open(self.output_annotation_path, "w") as f:
            json.dump(self.coco_annotations, f)

    def create_llm_dataset(self):
        self.coco = COCO(self.output_annotation_path)
        pattern = re.compile(r"(.+?_\d+_\d+)\..+")
        for image in self.coco.dataset["images"]:
            base_name = pattern.match(image["file_name"]).group(1)
            filename = image["file_name"]

            prompt = f"<image>\nCould you analyze this image and tell me the total number of"
            answer = f"There"

            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image["id"]))
            classes = {}
            for ann in anns:
                class_name = self.coco.loadCats(ann["category_id"])[0]["name"]
                if class_name not in classes:
                    classes[class_name] = 1
                else:
                    classes[class_name] += 1
            
            if len(classes) == 1:
                answer += " is"
            elif len(classes) > 1:
                answer += " are"
            else:
                continue

            for class_name, count in classes.items():
                prompt += f" {class_name},"
                if count > 1:
                    class_name += "s"
                answer += f" {count} {class_name},"
            
            prompt = prompt[:-1]+ "?"
            answer = answer[:-1] + "."

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

        output_path = os.path.join(self.data_dir, "llm_annotations.json")
        with open(output_path, "w") as f:
            json.dump(self.llm_annotations, f)
    
    def prepare_dataset(self):
        self.parse_dataset()
        self.create_llm_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset. Should contain images, bboxes and classes folders")
    parser.add_argument("--output_annotation", type=str, default="data", help="Annotation name to save the prepared dataset in COCO format")
    args = parser.parse_args()

    preparation = PrepareDataset(args)
    preparation.prepare_dataset()

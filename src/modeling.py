from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
import os
import json
import random
import multiprocessing

def fix_annotations(annotations, categories):
    """Fix and validate annotations"""
    # Ensure we have at least one category
    if not categories:
        categories = [{'id': 1, 'name': 'pest', 'supercategory': 'pest'}]
        print("Warning: No categories found, creating default category")
    
    fixed_annotations = []
    for ann in annotations:
        # Create a new annotation with all required fields
        fixed_ann = {
            'id': ann.get('id', len(fixed_annotations) + 1),
            'image_id': ann['image_id'],
            'category_id': ann.get('category_id', categories[0]['id']),
            'bbox': ann.get('bbox', [0, 0, 10, 10]),
            'area': ann.get('area', 100.0),
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', [[0, 0, 0, 10, 10, 10, 10, 0]])
        }
        fixed_annotations.append(fixed_ann)
    return fixed_annotations, categories

def split_coco_dataset(input_json, train_json, val_json, split_ratio=0.8):
    """Split COCO dataset into train and validation sets"""
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print("Dataset keys:", data.keys())
    print("Number of images:", len(data['images']))
    if 'annotations' in data:
        print("Number of annotations:", len(data['annotations']))
    
    # Ensure categories exist and are not empty
    if 'categories' not in data or not data['categories']:
        data['categories'] = [{'id': 1, 'name': 'pest', 'supercategory': 'pest'}]
        print("Created default category")
    print("\nCategories:", data['categories'])
    
    # Fix annotations
    if 'annotations' not in data:
        print("No annotations found, creating default annotations")
        data['annotations'] = []
        for img in data['images']:
            data['annotations'].append({
                'id': img['id'],
                'image_id': img['id'],
                'category_id': 1
            })
    
    # Fix and validate all annotations
    data['annotations'], data['categories'] = fix_annotations(data['annotations'], data['categories'])
    
    # Print sample annotation for debugging
    if data['annotations']:
        print("\nSample annotation after fixing:", data['annotations'][0])
    
    # Randomly split image IDs
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * split_ratio)
    train_ids = set(image_ids[:split_idx])
    
    # Split images and annotations
    train_images = [img for img in data['images'] if img['id'] in train_ids]
    val_images = [img for img in data['images'] if img['id'] not in train_ids]
    train_anns = [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
    val_anns = [ann for ann in data['annotations'] if ann['image_id'] not in train_ids]
    
    # Create train and val datasets
    train_data = {
        'images': train_images,
        'annotations': train_anns,
        'categories': data['categories']
    }
    
    val_data = {
        'images': val_images,
        'annotations': val_anns,
        'categories': data['categories']
    }
    
    # Save to files
    with open(train_json, 'w') as f:
        json.dump(train_data, f)
    with open(val_json, 'w') as f:
        json.dump(val_data, f)
    
    print(f"\nSplit complete:")
    print(f"Training: {len(train_images)} images, {len(train_anns)} annotations")
    print(f"Validation: {len(val_images)} images, {len(val_anns)} annotations")

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_and_train():
    # Directory paths
    dataset_dir = "./"
    input_json = os.path.join(dataset_dir, "PaddyPestData.json")
    train_json = os.path.join(dataset_dir, "train_annotations.json")
    val_json = os.path.join(dataset_dir, "val_annotations.json")
    images_dir = os.path.join(dataset_dir, "images")

    # Remove existing split files to force regeneration
    if os.path.exists(train_json):
        os.remove(train_json)
    if os.path.exists(val_json):
        os.remove(val_json)

    # Split dataset
    split_coco_dataset(input_json, train_json, val_json)

    # Register the datasets
    register_coco_instances("paddy_train", {}, train_json, images_dir)
    register_coco_instances("paddy_val", {}, val_json, images_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Model configs
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    
    # Dataset configs
    cfg.DATASETS.TRAIN = ("paddy_train",)
    cfg.DATASETS.TEST = ("paddy_val",)
    
    # Dataloader configs
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    
    # Solver configs
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    
    # Evaluation configs
    cfg.TEST.EVAL_PERIOD = 50  # Evaluate every 50 iterations
    
    # Set output directory
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create trainer
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    
    print("Starting training...")
    trainer.train()
    
    return cfg

if __name__ == "__main__":
    multiprocessing.freeze_support()
    setup_and_train()


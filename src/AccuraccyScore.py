from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from modeling import setup_and_train, CustomTrainer

# Get config and trainer
cfg = setup_and_train()
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=True)  # Load the trained model

evaluator = COCOEvaluator("paddy_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "paddy_val")
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print(metrics)

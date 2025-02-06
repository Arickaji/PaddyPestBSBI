import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Setup configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth")  # path to the model we just trained
cfg.MODEL.DEVICE = "cpu"

# Set up the predictor
predictor = DefaultPredictor(cfg)

#Test_images/00000504.jpg

dataset_dir = "./"  

# Load a test image
test_image_path = os.path.join(dataset_dir, "Test_images", "00000504.jpg")
image = cv2.imread(test_image_path)
outputs = predictor(image)

# Visualize the predictions
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("paddy_val"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

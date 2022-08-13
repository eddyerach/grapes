# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.utils.visualizer import ColorMode

class MLModel:
    def __init__(self,dir_path):
        self.cfg = BuildCfg(dir_path)
        self.dir_path = dir_path#"D:/Austral Falcon/"

        self.predictor = DefaultPredictor(self.cfg)
       

    #real_len: cantidad de uvas reales para verificacion
    def predict(self,img,real_len=-1):
        #print(file)    
        #im = cv2.imread(file)
        #real_len = len(d["annotations"])
        outputs = self.predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        pred_len = len(outputs["instances"])
        #print("REAL NUM:",real_len," PRED NUM:",pred_len)
                
        v = Visualizer(img[:, :, ::-1],
                #metadata=metadata, 
                scale=0.5, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_im = out.get_image()[:, :, ::-1]
        #cv2.imshow("",out_im)
        #cv2.waitKey(0)

        return {"detect":out_im,"real_count":real_len,"count":pred_len}






def BuildCfg(dir_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = dir_path#"D:/Austral Falcon/"
    cfg.MODEL.DEVICE = "cpu" #TODO: Para probar cpu, sacar esto
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    
    return cfg





if __name__ == "__main__":

    model = MLModel("/home/azureuser/grape/")
    predictions =[]
    images = glob.glob("dataset/*")

    for file in images:
        im = cv2.imread(file)
        pred = model.predict(im)
        pred["filename"] = file
        predictions.append(pred)
    
    print("Dirección de la imagen\tPredicción")
    for i in predictions:
        print(i["filename"],"\t\t",i["count"])

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog






from detectron2.data import MetadataCatalog, DatasetCatalog

def load_data(t="train"):
    if t == "train":
        with open("dataset.json", 'r') as file:
            train = json.load(file)["train"]
        return train
    elif t == "val":
      with open("dataset.json", 'r') as file:
          val = json.load(file)["val"]
    return val


for d in ["train", "val"]:
    DatasetCatalog.register(d, lambda d=d: load_data(d))
    MetadataCatalog.get(d).set(thing_classes=["Uva"])
metadata = MetadataCatalog.get("train")


with open("dataset.json", 'r') as file:
    train = json.load(file)["train"]
    #val = json.load(file)["val"]


for d in random.sample(train, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("",out.get_image()[:, :, ::-1])
    cv2.waitKey(0)


from detectron2.engine import DefaultTrainer

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
cfg.OUTPUT_DIR = "output"
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':

    
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    with open("dataset.json", 'r') as file:
        #train = json.load(file)["train"]
        val = json.load(file)["val"]

    predictor = DefaultPredictor(cfg)
    from detectron2.utils.visualizer import ColorMode



    for i in range(83):
        print(i)    
        im = cv2.imread("dataset/"+str(i)+".jpg")
        #real_len = len(d["annotations"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        #pred_len = len(outputs["instances"])
        #print("REAL NUM:",real_len," PRED NUM:",pred_len)
        
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("mask_rcnn_output/"+str(i)+".jpg",out.get_image()[:, :, ::-1])
        #cv2.imshow("",out.get_image()[:, :, ::-1])
        #cv2.waitKey(0)


'''
    for d in val:
        print(d["file_name"])    
        im = cv2.imread(d["file_name"])
        real_len = len(d["annotations"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        pred_len = len(outputs["instances"])
        print("REAL NUM:",real_len," PRED NUM:",pred_len)
        
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
'''
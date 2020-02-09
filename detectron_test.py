from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.modeling import build_model
import cv2
import os

if __name__ == '__main__':
    im_1 = cv2.imread(os.getcwd() + '/stimulis/026.png')
    im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2RGB)
    im_2 = cv2.imread(os.getcwd() + '/stimulis/021.png')
    im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2RGB)
    im_3 = cv2.imread(os.getcwd() + '/stimulis/051.png')
    im_3 = cv2.cvtColor(im_3, cv2.COLOR_BGR2RGB)
    im_4 = cv2.imread(os.getcwd() + '/stimulis/008.png')
    im_4 = cv2.cvtColor(im_4, cv2.COLOR_BGR2RGB)
    im_5 = cv2.imread(os.getcwd() + '/stimulis/068.png')
    im_5 = cv2.cvtColor(im_5, cv2.COLOR_BGR2RGB)
    im_6 = cv2.imread(os.getcwd() + '/stimulis/106.png')
    im_6 = cv2.cvtColor(im_6, cv2.COLOR_BGR2RGB)

    # Create config
    cfg = get_cfg()

    #cfg.merge_from_file(os.getcwd() + '/official_doc/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    #cfg.merge_from_file(os.getcwd() + '/official_doc/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    #cfg.merge_from_file(os.getcwd() + '/official_doc/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml')
    cfg.merge_from_file(os.getcwd() + '/official_doc/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml')

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    #cfg.MODEL.WEIGHTS = os.getcwd() + '/pretrained_models/faster_rcnn_r101_fpn.pkl'
    #cfg.MODEL.WEIGHTS = os.getcwd() + '/pretrained_models/mask_rcnn_r101_fpn.pkl'
    #cfg.MODEL.WEIGHTS = os.getcwd() + '/pretrained_models/keypoint_rcnn_r101_fpn.pkl'
    cfg.MODEL.WEIGHTS = os.getcwd() + '/pretrained_models/panoptic_r101_fpn.pkl'

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Make prediction
    pm_1 = predictor(im_1)
    pm_2 = predictor(im_2)
    pm_3 = predictor(im_3)
    pm_4 = predictor(im_4)
    pm_5 = predictor(im_5)
    pm_6 = predictor(im_6)

    v = Visualizer(im_5[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(pm_5["instances"].to("cpu"))
    cv2.imshow('', v.get_image())
    cv2.waitKey()

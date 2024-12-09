import argparse
import os
import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from bodyhands import *
from datasets import *
from bodyhands import add_bodyhands_config
from bodyhands import CustomVisualizer
import time

class CustomPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():  
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs], height, width)[0]
            return predictions

def prepareModel(cfg_file, weights, thresh):
    cfg = get_cfg()
    add_bodyhands_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.WEIGHTS = os.path.abspath(weights)
    predictor = CustomPredictor(cfg)
    return predictor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    parser.add_argument('--input', required=True, metavar='path to images', help='path to images')
    parser.add_argument('--thresh', required=False, metavar='threshold for hand detections', \
    	help='hand detection score threshold', default=0.7)
    parser.add_argument('--output', required=True, metavar='path to output', help='path to output')

    args = parser.parse_args()
    videos_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)
    if not os.path.exists(out_path):
        os.mkdir(out_path)   
    roi_score_thresh = float(args.thresh)
    model = prepareModel('./configs/BodyHands.yaml', './models/model.pth', roi_score_thresh)

    totalVideos = len(os.listdir(videos_path))
    count_video = 0
    
    
    
    for video in os.listdir(videos_path):
        count_video += 1
        print("Processing video # {}, total: {}".format(count_video, totalVideos))
        images = os.listdir(os.path.join(videos_path, video))
        count_image = 0
        os.makedirs(os.path.join(out_path, video), exist_ok=True)
        for image in images:
            count_image += 1
            print("Processing image # {}, total: {}".format(count_image, len(images)))
            im = cv2.imread(os.path.join(videos_path, video, image))
            height, width = im.shape[:2]
            ratio = width / height
            outheight = 256
            outwidth = int(ratio * outheight)
            im = cv2.resize(im, (outwidth, outheight))
            outputs = model(im)
            outputs = outputs["instances"].to("cpu")
            classes = outputs.pred_classes
            boxes = outputs.pred_boxes.tensor
            hand_indices = classes == 0
            hand_boxes = boxes[hand_indices]
            num_hands = hand_boxes.shape[0]
            with open(os.path.join(out_path, video,image.split('.')[0] + '.txt'), 'w') as f:
                for hand_no in range(num_hands):
                    box = hand_boxes[hand_no].view(-1).cpu().numpy()
                    xmin, ymin, xmax, ymax = box
                    f.write('hand' + str(hand_no) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')
        # im = cv2.imread(os.path.join(images_path, img))
        # height, width = im.shape[:2]
        # ratio = width / height
        # outheight = 256
        # outwidth = int(ratio * outheight)
        # im = cv2.resize(im, (outwidth, outheight))
        # outputs = model(im)
        # # v = CustomVisualizer(im[:, :, ::-1], MetadataCatalog.get("HandBodyContactHands_sub"), scale=1.0)
        # #######################################################################################
        # outputs = outputs["instances"].to("cpu")
        # classes = outputs.pred_classes
        # # body_ids = outputs.pred_body_ids
        # boxes = outputs.pred_boxes.tensor
        # # masks = outputs.pred_masks
        # hand_indices = classes == 0
        # # body_indices = classes == 1
        # hand_boxes = boxes[hand_indices]
        # # hand_masks = masks[hand_indices]
        # # hand_body_ids = body_ids[hand_indices]
        # # body_boxes = boxes[body_indices]
        # # body_body_ids = body_ids[body_indices]
        # num_hands = hand_boxes.shape[0]
        # body_masks = []
        # # for hand in range(num_hands):
        #     # print(hand_boxes[hand])
        # #     box = body_boxes[body_no].view(-1).cpu().numpy()
        # #     xmin, ymin, xmax, ymax = box
        # #     body_poly = [[(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]]
        # #     body_masks.append(body_poly)
        # print(time.time() - time_0)
        # if(count != 1):
        #     time_total += time.time() - time_0
        # with open(os.path.join(out_path, img.split('.')[0] + '.txt'), 'w') as f:
        #     for hand_no in range(num_hands):
        #         box = hand_boxes[hand_no].view(-1).cpu().numpy()
        #         xmin, ymin, xmax, ymax = box
        #         f.write('hand' + str(hand_no) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')
        
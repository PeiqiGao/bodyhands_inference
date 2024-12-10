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
from torch.utils.data import Dataset, DataLoader
import tqdm


class InferenceDataset(Dataset):
    def __init__(self, input_path, transform=None, device = torch.device('cuda')):
        self.input_path = input_path
        self.transform = transform
        self.video_name = []
        self.images_path = []
        self.images_name = []
        self.device = device
        for video in os.listdir(input_path):
            video_path = os.path.join(input_path, video)
            for img in os.listdir(video_path):
                if('img' not in img):
                    continue
                img_path = os.path.join(video_path, img)
                self.images_path.append(img_path)
                self.video_name.append(video)
                self.images_name.append(img)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        img = {'image': img.to(self.device), 'height': img.shape[1], 'width': img.shape[2]}
        return img, self.video_name[idx], self.images_name[idx]

def custom_collate_fn(batch):
    
    images = [item[0] for item in batch]
    video_name = [item[1] for item in batch]
    img_name = [item[2] for item in batch]
    
    return images,video_name,img_name

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Arguments for evaluation')
parser.add_argument('--input', required=True, metavar='path to images', help='path to images')
parser.add_argument('--thresh', required=False, metavar='threshold for hand detections', \
    help='hand detection score threshold', default=0.7)
parser.add_argument('--output', required=True, metavar='path to output', help='path to output')
parser.add_argument('--batch_size', required=True,type=int, metavar='batch size', help='batch size')

args = parser.parse_args()
out_path = args.output
batch_size = args.batch_size
if not os.path.exists(out_path):
    os.mkdir(out_path)   
    
roi_score_thresh = float(args.thresh)
# model = prepareModel('./configs/BodyHands.yaml', './models/model.pth', roi_score_thresh).to(device)

cfg = get_cfg()
add_bodyhands_config(cfg)
cfg.merge_from_file(cfg_filename='./configs/BodyHands.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_score_thresh
cfg.MODEL.WEIGHTS = './models/model.pth'
model = build_model(cfg).to(device)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)



dataset = InferenceDataset(args.input,device = device)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

length = len(dataloader)
index = 0
for batch in tqdm.tqdm(dataloader):
    index +=1
    img, video_name, img_name = batch
    with torch.no_grad():
        outputs = model(img,256,344)
    for i in range(len(outputs)):
        output = outputs[i]
        output = output['instances'].to('cpu')
        classes = output.pred_classes
        boxes = output.pred_boxes.tensor
        hand_indices = classes == 0
        hand_boxes = boxes[hand_indices]
        num_hands = hand_boxes.shape[0]
        os.makedirs(os.path.join(out_path, video_name[i]), exist_ok=True)
        with open(os.path.join(out_path, video_name[i],img_name[i][:-4]+'.txt'), 'w') as f:
            for hand_no in range(num_hands):
                box = hand_boxes[hand_no].view(-1).cpu().numpy()
                xmin, ymin, xmax, ymax = box
                f.write('hand' + str(hand_no) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')
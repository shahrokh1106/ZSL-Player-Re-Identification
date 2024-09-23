import sys
sys.path.append("./yolov9main")
import cv2
import torch
import glob
import os
from pathlib import Path
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages
from utils.general import check_img_size,non_max_suppression,Profile,scale_boxes
from utils.segment.general import process_mask
import numpy as np


def get_yolo9_model(imgsz):
    device=select_device(0) 
    data="./yolov9main/models/segment/gelan-c-seg.yaml"
    dnn=False  # use OpenCV DNN for ONNX inference
    half=False  # use FP16 half-precision inference
    weights= './yolov9main/gelan-c-seg.pt' 
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1, 3, *imgsz)) 
    return model

def get_reid_dataset_paths(reid_dataset_path):
    dataset_dict = {}
    main_paths = glob.glob(os.path.join(reid_dataset_path, "*"))
    for index in range(len(main_paths)):
        sample_paths = glob.glob(os.path.join(main_paths[index], "*"))
        anchor_folder_path, negative_folder_path, positive_folder_path = sample_paths[0],sample_paths[1],sample_paths[2]
        anchor_img_path = glob.glob(os.path.join(anchor_folder_path, "*"))
        negative_img_paths = glob.glob(os.path.join(negative_folder_path, "*"))
        positive_img_path = glob.glob(os.path.join(positive_folder_path, "*"))
        sample_paths = anchor_img_path+negative_img_paths+positive_img_path
        dataset_dict.update({os.path.basename(main_paths[index]): sample_paths})
    return  dataset_dict

if __name__ == "__main__":
    imgsz=(640, 640)
    model = get_yolo9_model(imgsz)
    reid_dataset_path = "./datasets/ReidDataset_Rugby" # or "./datasets/ReidDataset_Netball"
    out_put_path = os.path.join(os.getcwd(),"Rugby_Masks")
    reid_dataset = get_reid_dataset_paths(reid_dataset_path)

    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    classes=None
    agnostic_nms=False  # class-agnostic NMS
    max_det=1000  # maximum detections per image

    pad_offset = 100 #padding is to make sure that Yolo detects the player completely for the segmentation purposes
    for key in list(reid_dataset.keys()):
        image_paths = reid_dataset[key]
        image_list = []
        image_list_bgr = []
        image_pad_sizes = []
        for image_path in image_paths:
            img_org = cv2.imread(image_path)
            image_list_bgr.append(img_org)
            img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.copyMakeBorder(img_rgb, pad_offset, pad_offset, pad_offset, pad_offset,  cv2.BORDER_REPLICATE).astype(np.uint8)
            image_pad_sizes.append((img_rgb.shape[0],img_rgb.shape[1]))
            img_rgb = cv2.resize(img_rgb, imgsz)
            img_rgb = np.transpose(img_rgb, (2, 0, 1))
            image_list.append(img_rgb)
        ims = np.stack(image_list)
        ims = torch.from_numpy(ims)
        ims = (ims.float()/255).to(model.device)
        predd, proto = model(ims,augment=False, visualize=False)[:2]
        pred = non_max_suppression(predd, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        mask_list = []
        for i, det in enumerate(pred):
            if len(det):
                masks = process_mask(proto[2].squeeze(0)[i], det[:, 6:], det[:, :4], ims.shape[2:], upsample=True)  # HWC
                msk = masks.cpu().numpy()[0].astype(np.uint8)
                msk = cv2.resize(msk, (image_pad_sizes[i][1],image_pad_sizes[i][0]))
                msk = msk[pad_offset:msk.shape[0]-pad_offset, pad_offset:msk.shape[1]-pad_offset]
                mask_list.append(msk)
            else:
                msk=np.zeros((640,640)).astype(np.uint8)
                msk = cv2.resize(msk, (image_pad_sizes[i][1],image_pad_sizes[i][0]))
                msk = msk[pad_offset:msk.shape[0]-pad_offset, pad_offset:msk.shape[1]-pad_offset]
                mask_list.append(msk)
                                
        if not os.path.exists(out_put_path):
            os.makedirs(out_put_path)
        sample_mask_path = os.path.join(out_put_path,key)
        anchor_mask_path = os.path.join(sample_mask_path,"anchor")
        if not os.path.exists(anchor_mask_path):
            os.makedirs(anchor_mask_path)
        positive_mask_path = os.path.join(sample_mask_path,"positive")
        if not os.path.exists(positive_mask_path):
            os.makedirs(positive_mask_path)
        negative_mask_path = os.path.join(sample_mask_path,"negative")
        if not os.path.exists(negative_mask_path):
            os.makedirs(negative_mask_path)

        cv2.imwrite(os.path.join(anchor_mask_path,os.path.basename(image_paths[0])[:-4]+".png"), (mask_list[0]*255).astype(np.uint8))
        cv2.imwrite(os.path.join(positive_mask_path,os.path.basename(image_paths[10])[:-4]+".png"), (mask_list[10]*255).astype(np.uint8))
        for i in range(1,10):
            cv2.imwrite(os.path.join(negative_mask_path,os.path.basename(image_paths[i])[:-4]+".png"), (mask_list[i]*255).astype(np.uint8))
 

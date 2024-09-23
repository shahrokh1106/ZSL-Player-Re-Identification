import numpy as np
import cv2
import glob
import os

def refine_mask_gui(img_path,msk_path):
    def update_mask(event, x, y, flags, param):
        global drawing, mask, color, radius
        if event == cv2.EVENT_LBUTTONDOWN:  
            drawing = True
            color = 255  
        elif event == cv2.EVENT_RBUTTONDOWN:  
            drawing = True
            color = 0  
        elif event == cv2.EVENT_MOUSEMOVE:  
            if drawing:
                cv2.circle(mask, (x, y), radius, color, -1)  
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:  # Stop drawing
            drawing = False
    global drawing, mask,radius
    image = cv2.imread(img_path)  
    mask = cv2.imread(msk_path, 0) 
    org_size = mask.shape
    show_size = (640,320)
    image = cv2.resize(image, (show_size[1],show_size[0]))
    mask = cv2.resize(mask, (show_size[1],show_size[0]))
    cv2.namedWindow('Image with Mask')
    cv2.setMouseCallback('Image with Mask', update_mask)
    drawing = False 
    alpha = 0.8 
    radius = 13
    while True:
        color_mask = cv2.merge([mask, mask, mask])
        combined = cv2.addWeighted(image, alpha, color_mask, 0.3, 0)
        cv2.imshow('Image with Mask', combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('1'):
            alpha+=0.03
            if alpha>1:
                alpha=1
        if key == ord('2'):
            alpha-=0.03
            if alpha<0:
                alpha=0
        if key == ord('='):
            radius+=2
            if radius>20:
                radius=20
        if key == ord('-'):
            radius-=2
            if radius<1:
                radius=1

    cv2.destroyAllWindows()
    mask = cv2.resize(mask, (org_size[1],org_size[0]))
    return mask

def refine_masks(img_dataset_path, msk_dataset_path, msk_dataset_path_output):
    if not os.path.exists(msk_dataset_path_output):
        os.makedirs(msk_dataset_path_output)
    img_dataset_folder_paths = glob.glob(os.path.join(img_dataset_path, "*"))
    for INDEX in range(0,len(img_dataset_folder_paths)):
        selected_img_folder_path = img_dataset_folder_paths[INDEX]
        selected_mask_folder_path = os.path.join(msk_dataset_path,os.path.basename(selected_img_folder_path))
        anchor_img_path = glob.glob(os.path.join(selected_img_folder_path,"anchor","*"))[0]
        positive_img_path = glob.glob(os.path.join(selected_img_folder_path,"positive","*"))[0]
        negative_img_paths = glob.glob(os.path.join(selected_img_folder_path,"negative","*"))
        anchor_msk_path = glob.glob(os.path.join(selected_mask_folder_path,"anchor","*"))[0]
        positive_msk_path = glob.glob(os.path.join(selected_mask_folder_path,"positive","*"))[0]
        negative_msk_paths = glob.glob(os.path.join(selected_mask_folder_path,"negative","*"))
        
        refined_anchor_mask = refine_mask_gui(anchor_img_path,anchor_msk_path)
        refined_positive_mask = refine_mask_gui(positive_img_path,positive_msk_path)
        refined_negative_masks = []
        for i,negative_img_path in enumerate(negative_img_paths):
            refined_negative_masks.append(refine_mask_gui(negative_img_path,negative_msk_paths[i]))
        
        selected_refined_mask_folder_path=os.path.join(msk_dataset_path_output,os.path.basename(selected_img_folder_path))
        if not os.path.exists(selected_refined_mask_folder_path):
            os.makedirs(selected_refined_mask_folder_path)
        
        anchor_refined_msk_path=os.path.join(selected_refined_mask_folder_path,"anchor")
        if not os.path.exists(anchor_refined_msk_path):
            os.makedirs(anchor_refined_msk_path)
        cv2.imwrite(os.path.join(anchor_refined_msk_path,os.path.basename(anchor_img_path)[:-4]+".png"), refined_anchor_mask)
        
        positive_refined_msk_path=os.path.join(selected_refined_mask_folder_path,"positive")
        if not os.path.exists(positive_refined_msk_path):
            os.makedirs(positive_refined_msk_path)
        cv2.imwrite(os.path.join(positive_refined_msk_path,os.path.basename(positive_img_path)[:-4]+".png"), refined_positive_mask)
        
        negative_refined_msk_path=os.path.join(selected_refined_mask_folder_path,"negative")
        if not os.path.exists(negative_refined_msk_path):
            os.makedirs(negative_refined_msk_path)
        
        for i, refined_negative_mask in enumerate(refined_negative_masks):
            cv2.imwrite(os.path.join(negative_refined_msk_path,os.path.basename(negative_img_paths[i])[:-4]+".png"), refined_negative_mask)


def get_masked_images(img_dataset_path,msk_dataset_path,msk_dataset_path_output):
    def get_roi(img_path,msk_path):
        img = cv2.imread(img_path)
        msk = cv2.imread(msk_path,0)
        msk = cv2.erode(msk, np.ones((5, 5), np.uint8), iterations=1)
        img[msk==0]=(0,0,0)
        return img
    if not os.path.exists(msk_dataset_path_output):
        os.makedirs(msk_dataset_path_output)
    img_dataset_folder_paths = glob.glob(os.path.join(img_dataset_path, "*"))
    for INDEX in range(0,len(img_dataset_folder_paths)):
        selected_img_folder_path = img_dataset_folder_paths[INDEX]
        selected_mask_folder_path = os.path.join(msk_dataset_path,os.path.basename(selected_img_folder_path))
        anchor_img_path = glob.glob(os.path.join(selected_img_folder_path,"anchor","*"))[0]
        positive_img_path = glob.glob(os.path.join(selected_img_folder_path,"positive","*"))[0]
        negative_img_paths = glob.glob(os.path.join(selected_img_folder_path,"negative","*"))
        anchor_msk_path = glob.glob(os.path.join(selected_mask_folder_path,"anchor","*"))[0]
        positive_msk_path = glob.glob(os.path.join(selected_mask_folder_path,"positive","*"))[0]
        negative_msk_paths = glob.glob(os.path.join(selected_mask_folder_path,"negative","*"))

        
        refined_anchor_mask = get_roi(anchor_img_path,anchor_msk_path)
        refined_positive_mask = get_roi(positive_img_path,positive_msk_path)
        refined_negative_masks = []
        for i,negative_img_path in enumerate(negative_img_paths):
            refined_negative_masks.append(get_roi(negative_img_path,negative_msk_paths[i]))
        
        selected_refined_mask_folder_path=os.path.join(msk_dataset_path_output,os.path.basename(selected_img_folder_path))
        if not os.path.exists(selected_refined_mask_folder_path):
            os.makedirs(selected_refined_mask_folder_path)
        
        anchor_refined_msk_path=os.path.join(selected_refined_mask_folder_path,"anchor")
        if not os.path.exists(anchor_refined_msk_path):
            os.makedirs(anchor_refined_msk_path)
        cv2.imwrite(os.path.join(anchor_refined_msk_path,os.path.basename(anchor_img_path)[:-4]+".jpg"), refined_anchor_mask)
        
        positive_refined_msk_path=os.path.join(selected_refined_mask_folder_path,"positive")
        if not os.path.exists(positive_refined_msk_path):
            os.makedirs(positive_refined_msk_path)
        cv2.imwrite(os.path.join(positive_refined_msk_path,os.path.basename(positive_img_path)[:-4]+".jpg"), refined_positive_mask)
        
        negative_refined_msk_path=os.path.join(selected_refined_mask_folder_path,"negative")
        if not os.path.exists(negative_refined_msk_path):
            os.makedirs(negative_refined_msk_path)
        
        for i, refined_negative_mask in enumerate(refined_negative_masks):
            cv2.imwrite(os.path.join(negative_refined_msk_path,os.path.basename(negative_img_paths[i])[:-4]+".jpg"), refined_negative_mask)    


    

import os
import random
import numpy as np
import glob
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# use this function to load the dataset
def load_reid_dataset (destination):
    reid_dataset = []
    for sample in glob.glob(os.path.join(destination, "*")):
        anchor_path = glob.glob(os.path.join(sample, "*"))[0]
        positive_path = glob.glob(os.path.join(sample, "*"))[2]
        negative_paths = glob.glob(os.path.join(sample, "*"))[1]
        anchoar_img_path = glob.glob(os.path.join(anchor_path, "*"))[0]    
        positive_img_path = glob.glob(os.path.join(positive_path, "*"))[0]
        negative_img_paths = glob.glob(os.path.join(negative_paths, "*"))
        reid_dataset.append([anchoar_img_path]+negative_img_paths+[positive_img_path])  
    random.seed(45)
    random.shuffle(reid_dataset)
    return reid_dataset

def GetScores (reid_feature_dataset,dataset_name, model_name):
    def compute_distances(query, gallery, metric='cosine'):
        if metric == 'cosine':
            similarities = cosine_similarity([query], gallery)[0]
            distances = 1 - similarities  # Convert similarity to distance
        else:  # Euclidean distance
            distances = np.linalg.norm(gallery - query, axis=1)
        return distances
    
    def get_top_k(distances, k):
        return np.argsort(distances)[:k]
    
    def compute_ap(ranked_indices, positive_idx):
        try:
            rank = np.where(ranked_indices == positive_idx)[0][0] + 1  # Rank starts at 1
            return 1.0 / rank
        except IndexError:
            return 0.0  # If positive_idx is not found in ranked_indices
    

    top_1_results = []
    top_3_results = []
    top_5_results = []
    aps = []
    
    for sample in reid_feature_dataset:
        query = sample[0]
        gallery = sample[1:]
        positive_idx = 9 
    
        distances = compute_distances(query, gallery)
        
        top_1_idx = get_top_k(distances, 1)
        top_3_idx = get_top_k(distances, 3)
        top_5_idx = get_top_k(distances, 5)
    
        top_1_results.append(positive_idx in top_1_idx)
        top_3_results.append(positive_idx in top_3_idx)
        top_5_results.append(positive_idx in top_5_idx)
    
        ranked_indices = np.argsort(distances)
        ap = compute_ap(ranked_indices, positive_idx)
        aps.append(ap)
    
        
    overal_top_1_percentage  = (np.sum(top_1_results)/ len(reid_feature_dataset))*100
    overal_top_3_percentage  = (np.sum(top_3_results)/ len(reid_feature_dataset))*100
    overal_top_5_percentage  = (np.sum(top_5_results)/ len(reid_feature_dataset))*100
    mAP = np.mean(aps) * 100

    results_dict = {"Top-1": overal_top_1_percentage,
                    "Top-3": overal_top_3_percentage,
                    "Top-5": overal_top_5_percentage,
                    "mAP" : mAP
    }
    df = pd.DataFrame([results_dict])
    out_path = "REID/results_csv/"+dataset_name
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    df.to_csv(os.path.join(out_path,model_name+".csv") , index=False)



def main(reid_dataset, dataset_name):
    # Getting the features-->out_features
    pass
    # GetScores (out_features,dataset_name, model_name = "BPBreID")

if __name__ == "__main__":
    dataset_names = ["ReidDataset_Rugby", "ReidDataset_Rugby_Masked", "ReidDataset_Netball", "ReidDataset_Netball_Masked"]
    for dataset_name in dataset_names:
        reid_dataset = load_reid_dataset (destination = "REID/datasets/"+dataset_name)  
        main(reid_dataset, dataset_name)
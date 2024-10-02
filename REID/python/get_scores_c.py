import os
import random
import sys
import argparse
import torch
import torchreid


MAIN_PATH = os.path.join("REID","bpbreid")
sys.path.append(MAIN_PATH)
sys.path.insert(0, os.path.abspath("REID/bpbreid"))


import torchreid.scripts.main as bpbreid_main
from torchreid.tools.feature_extractor import FeatureExtractor
#from torchreid.getFeatures import extract
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        '--save_dir', type=str, default='', help='path to output root dir'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    parser.add_argument(
        '--job-id',
        type=int,
        default=None,
        help='Slurm job id'
    )
    parser.add_argument(
        '--inference-enabled',
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    cfg = bpbreid_main.build_config(args, args.config_file)

    engine, model = bpbreid_main.build_torchreid_model_engine(cfg)

    print(dataset_name)

    extractor = FeatureExtractor(
        cfg,
        device='cpu',
        model = model
    )

    #outTensor =[]
    out = []
    proc =0
    for item in reid_dataset:
        print(proc)
        proc=proc+1

        result = extractor(item)
        #outTensor.append(result)
        parts = result[0]['parts']
        parts_reshaped = parts.view(11, -1)

        foreg= result[0]['foreg']
        #out.append(foreg.numpy())

        r = torch.cat((parts_reshaped, foreg), dim=1)

        out.append(r.numpy())

    GetScores(out, dataset_name, model_name="BPBreID")


    print("Finished dataset")
    #pass
    # GetScores (out_features,dataset_name, model_name = "BPBreID")

if __name__ == "__main__":
    dataset_names = ["ReidDataset_Rugby", "ReidDataset_Rugby_Masked", "ReidDataset_Netball", "ReidDataset_Netball_Masked"]
    for dataset_name in dataset_names:
        reid_dataset = load_reid_dataset (destination = "REID/datasets/"+dataset_name) #gibran, changed destination = "REID/datasets/"+dataset_name  
        main(reid_dataset, dataset_name)
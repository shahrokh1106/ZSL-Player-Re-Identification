from torchreid.tools.feature_extractor import FeatureExtractor
import random
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch


def load_reid_dataset(destination="C:/Users/kamehouse/Desktop/repo/ReidDataset"):
    reid_dataset = []
    for sample in glob.glob(os.path.join(destination, "*")):
        anchor_path = glob.glob(os.path.join(sample, "*"))[0]
        positive_path = glob.glob(os.path.join(sample, "*"))[2]
        negative_paths = glob.glob(os.path.join(sample, "*"))[1]
        anchoar_img_path = glob.glob(os.path.join(anchor_path, "*"))[0]
        positive_img_path = glob.glob(os.path.join(positive_path, "*"))[0]
        negative_img_paths = glob.glob(os.path.join(negative_paths, "*"))
        reid_dataset.append([anchoar_img_path] + negative_img_paths + [positive_img_path])
    random.seed(45)
    random.shuffle(reid_dataset)
    return reid_dataset

def GetScores(reid_feature_dataset, model_name="model_name"):
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

    def compute_cmc(ranked_indices, positive_idx, k_values):
        cmc = np.zeros(len(k_values))
        rank = np.where(ranked_indices == positive_idx)[0][0] + 1
        for i, k in enumerate(k_values):
            if rank <= k:
                cmc[i] = 1
        return cmc

    top_1_results = []
    top_3_results = []
    top_5_results = []
    aps = []
    cmcs = []

    j=0
    for sample in reid_feature_dataset:
       #print("sample: ", j)
        #j=j+1
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

        cmc = compute_cmc(ranked_indices, positive_idx, [1, 3, 5])
        cmcs.append(cmc)

    overal_top_1_percentage = (np.sum(top_1_results) / len(reid_feature_dataset)) * 100
    overal_top_3_percentage = (np.sum(top_3_results) / len(reid_feature_dataset)) * 100
    overal_top_5_percentage = (np.sum(top_5_results) / len(reid_feature_dataset)) * 100
    mAP = np.mean(aps) * 100
    CMC1, CMC3, CMC5 = np.mean(cmcs, axis=0) * 100

    results_dict = {"Top-1": overal_top_1_percentage,
                    "Top-3": overal_top_3_percentage,
                    "Top-5": overal_top_5_percentage,
                    "mAP": mAP,
                    "CMC-1": CMC1,
                    "CMC-3": CMC3,
                    "CMC-5": CMC5
                    }
    df = pd.DataFrame([results_dict])
    df.to_csv('c:/Users/kamehouse/Desktop/repo/' + model_name + ".csv", index=False)


def extract(cfg, model):
    extractor = FeatureExtractor(
        cfg,
        device='cpu',
        model = model
    )

    #data = load_reid_dataset(destination="C:/Users/gba2_/Desktop/repo/market")
    #data = load_reid_dataset(destination="C:/Users/gba2_/Desktop/repo/ReidDataset_short")
    data = load_reid_dataset()

    #outTensor =[]
    out = []
    proc =0
    for item in data:
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

    GetScores(out)

    print("Scores obtained")

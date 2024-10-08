# Exploring the Application of Knowledge Transfer to Sports Video Data

This is the official repository for the paper titled "Exploring the Application of Knowledge Transfer to Sports Video Data", exploring Zero Shot Learning (ZSL) for Player Re-Identification and Action Regocnition on Rugby League and Netball broadcast video datasets. 

## Re-Identification
The "REID" folder contains the prepared Re-ID datasets along with the Python scripts used to generate the results presented in the paper. 
### DATASETS
We provide two datasets: one for [Netball](REID/datasets/ReidDataset_Netball) and one for [Rugby League](REID/datasets/ReidDataset_Rugby). To further investigate the ZSL approach, we extended these datasets by creating masked versions ([Masked Netball](REID/datasets/ReidDataset_Netball_Masked) and [Masked Rugby](REID/datasets/ReidDataset_Rugby_Masked)), where the backgrounds have been made black to let the models focus solely on the players.

In each dataset, there are 100 samples, each consisting of a query image paired with a gallery set of ten images. Within each gallery set, nine images represent negative matches, while one image is positive. To create a challenging environment for evaluating ZSL re-identification approaches, we carefully selected negative matches where the players performed the same or similar actions and wore jerseys of similar colors to the query player. We used images of the same individual performing different actions for the positive match, ensuring that the re-identification models are rigorously tested under varied and demanding conditions.

![reid_dataset_samples.png](REID/figs/reid_dataset_samples.png)


We used YOLO-v9 to generate the initial player masks, and then manually refined them to correct inaccuracies, especially in cases where more than one player appeared in a single patch. To test YOLO-v9 on the dataset, please first follow the instructions in [YOLO-v9 REDAME](yolo9main/README.md) file to install dependencies, and then run [`get_masks.py`](REID/python/get_masks.py) to get the initial masks. We have also provied the scripts that we used to refine the masks in [`utilities.py`](REID/python/utilities.py).
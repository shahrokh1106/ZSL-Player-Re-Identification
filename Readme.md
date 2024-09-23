# Exploring the Application of Knowledge Transfer to Sports Video Data

This is the official repository for the paper titled "Exploring the Application of Knowledge Transfer to Sports Video Data", exploring Zero Shot Learning (ZSL) for Player Re-Identification and Action Regocnition on Rugby League and Netball broadcast video datasets. 

## Re-Identification
The "REID" folder contains the prepared Re-ID datasets along with the Python scripts used to generate the results presented in the paper. We provide two datasets: one for [Netball](REID/datasets/ReidDataset_Netball) and one for [Rugby League](REID/datasets/ReidDataset_Rugby). To further investigate the ZSL approach, we extended these datasets by creating masked versions ([Masked Netball](REID/datasets/ReidDataset_Netball_Masked) and [Masked Rugby](REID/datasets/ReidDataset_Rugby_Masked)), where the backgrounds have been made black to let the models focus solely on the players.


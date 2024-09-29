import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def darken_color(hex_color, factor=0.8):
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Darken the color
    darker_rgb = tuple(int(c * factor) for c in rgb)
    
    # Convert RGB back to hex
    darker_hex = '#{:02x}{:02x}{:02x}'.format(*darker_rgb)
    
    return darker_hex

def lighten_color(hex_color, factor=1.2):
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Lighten the color
    lighter_rgb = tuple(min(int(c * factor), 255) for c in rgb)
    
    # Convert RGB back to hex
    lighter_hex = '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)
    
    return lighter_hex

def main(sport_name = "rugby"):

    score_csv_root = os.path.join("REID", "results_csv")
    dataset_names_rugby = ["ReidDataset_Rugby", "ReidDataset_Rugby_Masked"]
    dataset_names_netball = ["ReidDataset_Netball", "ReidDataset_Netball_Masked"]

    if sport_name=="rugby":
        dataset_names = dataset_names_rugby
    else:
        dataset_names = dataset_names_netball


    model_names = ['MuDeep', 'HACNN', 'PCB', 'MLFN',
    'OSNet', 'OSNet-AIN','BPBreID', " ",'ResNet-5',
    'OSNet-soccer', 'DeiT-Tiny', 'ViT-B', 'ViT-L',"PRTreID"]

    dataframes = []
    dataframes_masked = []
    for model_name in model_names:
        if model_name!=" ":
            csv = pd.read_csv(os.path.join(score_csv_root, dataset_names[0], model_name+".csv" ))
            csv_masked = pd.read_csv(os.path.join(score_csv_root, dataset_names[1], model_name+".csv"))
            dataframes.append(csv[["Top-1","Top-3","Top-5", "mAP"]])
            dataframes_masked.append(csv_masked[["Top-1","Top-3","Top-5", "mAP"]])
        else:
            # this is to make a gap between the two groups on the plot
            dataframes.append(pd.DataFrame([{"Top-1":0,"Top-3":0, "Top-5":0,"mAP":0 }]))
            dataframes_masked.append(pd.DataFrame([{"Top-1":0,"Top-3":0, "Top-5":0,"mAP":0 }]))
        

    df = pd.concat(dataframes)
    df_masked = pd.concat(dataframes_masked)

    bar1 = list(df["Top-1"])
    bar2 = list(df["Top-3"])
    bar3 = list(df["Top-5"])
    bar4 = list(df["mAP"])
    line1 = list(df_masked["Top-1"])
    line2 = list(df_masked["Top-3"])
    line3 = list(df_masked["Top-5"])
    line4 = list(df_masked["mAP"])

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.axhline(y=40, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.axhline(y=60, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.axhline(y=80, color='gray', linestyle='--', linewidth=0.5, zorder=0)


    colors = ["#007fc5","#db7100","#00c0a2","#a9006a"]
    colors = ["#ffad2a","#1757dc","#2c6b00","#c10020"]

    # Bars for unmasked data (solid colors)
    light_factor = 1.2
    adjusted_width = 0.85 * width 
    rects1 = ax.bar(x - 1.5*width, bar1, adjusted_width, label='Top-1', color=lighten_color(colors[0], factor=light_factor))
    rects2 = ax.bar(x - 0.5*width, bar2, adjusted_width, label='Top-3', color=lighten_color(colors[1], factor=light_factor))
    rects3 = ax.bar(x + 0.5*width, bar3, adjusted_width, label='Top-5', color=lighten_color(colors[2], factor=light_factor))
    rects4 = ax.bar(x + 1.5*width, bar4, adjusted_width, label='mAP', color=lighten_color(colors[3], factor=light_factor))

    # Bars for masked data (dark colors)
    dark_factor = 0.7
    rects5 = ax.bar(x - 1.5*width, line1, adjusted_width, color=darken_color(colors[0], factor=dark_factor),label='Top-1 (masked)')
    rects6 = ax.bar(x - 0.5*width, line2, adjusted_width, color=darken_color(colors[1], factor=dark_factor),label='Top-3 (masked)')
    rects7 = ax.bar(x + 0.5*width, line3, adjusted_width, color=darken_color(colors[2], factor=dark_factor),label='Top-5 (masked)')
    rects8 = ax.bar(x + 1.5*width, line4, adjusted_width, color=darken_color(colors[3], factor=dark_factor),label='mAP (masked)')

    # Labels and legends
    ax.set_xlabel('Model Names', labelpad=10, fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14)
    # custom_patch = Patch(edgecolor='black', facecolor='none', label='Masked')
    ax.legend(handles=[rects1, rects2,rects3,rects4, rects5,rects6,rects7,rects8], loc='upper left',ncol=2,fontsize=13)
    plt.xticks(rotation=40, ha='right')

    # Group labels
    group_labels = ['Person ReID Models', 'Player ReID Models']
    group_positions = [(0, 6), (8, 13)]

    for (start, end), label in zip(group_positions, group_labels):
        mid = (start + end) / 2
        plt.plot([start-0.5, end+0.5], [-0.4, -0.4], color='black', lw=9.5, clip_on=False)
        ax.text(mid, -0.3, label, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))

    if sport_name=="rugby":
        img = mpimg.imread(os.path.join('REID','figs','rugby_sample.jpg'))
        img_masked = mpimg.imread(os.path.join('REID','figs','rugby_sample_masked.jpg'))
        imagebox = OffsetImage(img, zoom=0.65) 
        imagebox_masked = OffsetImage(img_masked, zoom=0.65) 
        ax.add_artist(AnnotationBbox(imagebox, (5.9,74), frameon=True))
        ax.add_artist(AnnotationBbox(imagebox_masked, (7.1,74), frameon=True))
    else:
        img = mpimg.imread(os.path.join('REID','figs','netball_sample.jpg'))
        img_masked = mpimg.imread(os.path.join('REID','figs','netball_sample_masked.jpg'))
        imagebox = OffsetImage(img, zoom=0.25) 
        imagebox_masked = OffsetImage(img_masked, zoom=0.25) 
        ax.add_artist(AnnotationBbox(imagebox, (5.9,75), frameon=True))
        ax.add_artist(AnnotationBbox(imagebox_masked, (7.1,75), frameon=True))

    # Adjust limits
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig("REID/figs/scores_"+sport_name+".png")
    plt.close()


if __name__ == "__main__":
    main(sport_name = "rugby")
    main(sport_name = "netball")
    
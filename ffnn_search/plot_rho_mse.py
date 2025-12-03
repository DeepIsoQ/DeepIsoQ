#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_path = "results/arch_search_results_stage2.csv"

def main():
    #load results
    df = pd.read_csv(results_path)

    #sort by MSE (best to worst)
    df = df.sort_values("val_mse", ascending = True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 25,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.dpi": 300
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)

    # Validation MSE
    sns.barplot(
        ax=axes[0],
        data=df,
        y="name",
        x="val_mse",
        palette="Blues_r",
        edgecolor="black"
    )
    axes[0].set_title("Validation MSE per Model")
    axes[0].set_xlabel("Validation MSE")
    axes[0].set_ylabel("Model")

    for i, v in enumerate(df["val_mse"]):
        axes[0].text(v, i, f"{v:.3f}", va="center", ha="left", fontsize=10)

    # Validation Pearson r
    df_pearson = df.sort_values("val_pearson", ascending=False).reset_index(drop=True)

    sns.barplot(
        ax=axes[1],
        data=df_pearson,
        y="name",
        x="val_pearson",
        palette="Greens_r",
        edgecolor="black"
    )
    axes[1].set_title("Validation Pearson Correlation per Model")
    axes[1].set_xlabel("Val Pearson r")
    axes[1].set_ylabel("Model")

    for i, v in enumerate(df_pearson["val_pearson"]):
        axes[1].text(v, i, f"{v:.3f}", va="center", ha="left", fontsize=10)

    out_path = "results/stage2_val_MSE_Pearson.png"
    fig.savefig(out_path, bbox_inches="tight")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(path_):
    save_path = ["ndvi_in_ROI.csv", "nirv_in_ROI.csv",
                 "fcvi_in_ROI.csv", "sif_in_ROI.csv",
                 "APAR_in_ROI.csv"]
    fig_format = '.png'
    fig_path = ["ndvi_in_ROI", "nirv_in_ROI",
                "fcvi_in_ROI", r"sif_in_ROI",
                "APAR_in_ROI"]
    Y = ['NDVI', 'NIRv', 'FCVI', 'SIF', 'APAR']

    # Plot
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rc('font', family='Times New Roman', size=10)

    for i in range(len(save_path)):
        df = pd.read_csv(os.path.join(path_, "VIs", save_path[i]), index_col=0)
        fig, ax = plt.subplots(constrained_layout=1, figsize=(8, 4), dpi=300)
        P_DICT = {'w1s1': '#CFE1F2', 'w1s2': '#CFE1F2', 'w1s3': '#CFE1F2', 'w1s4': '#CFE1F2', 'w1s5': '#CFE1F2',
                  'w2s1': '#93C4DE', 'w2s2': '#93C4DE', 'w2s3': '#93C4DE', 'w2s4': '#93C4DE', 'w2s5': '#93C4DE',
                  'w3s1': '#4A97C9', 'w3s2': '#4A97C9', 'w3s3': '#4A97C9', 'w3s4': '#4A97C9', 'w3s5': '#4A97C9',
                  'w4s1': '#1764AB', 'w4s2': '#1764AB', 'w4s3': '#1764AB', 'w4s4': '#1764AB', 'w4s5': '#1764AB'}
        sns.violinplot(data=df, palette=P_DICT, ax=ax)
        plt.ylabel(Y[i])
        fig.savefig(os.path.join(path_, "VIs", fig_path[i] + fig_format))


if __name__ == '__main__':
    path = r'F:\2022_8_20_sunny'
    main(path)

# gan_t_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gan_transformer_results(df_path, model_name='GAN-Transformer', save=False, save_dir='figures'):
    # 데이터 불러오기
    df = pd.read_csv(df_path)

    # truth, predicted 추출
    truth = df['truth'].values
    predicted = df['predicted'].values

    # Line Plot: 예측 vs 실제
    plt.figure(figsize=(14, 6))
    plt.plot(truth, label='Actual', linewidth=2)
    plt.plot(predicted, label='Predicted', linestyle='--')
    plt.title(f'{model_name} Prediction vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('CPU Usage (Scaled)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{model_name}_prediction_plot.png'))
    else:
        plt.show()

    # Error Distribution Plot
    errors = np.array(truth) - np.array(predicted)
    plt.figure(figsize=(10, 4))
    plt.hist(errors, bins=40, alpha=0.7, color='orange', edgecolor='black')
    plt.title(f'{model_name} Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(save_dir, f'{model_name}_error_distribution.png'))
    else:
        plt.show()

if __name__ == "__main__":
    dataset_name = 'custom_cpu_dataset'
    df_path = f'RTX2080/prediction-outputs/{dataset_name}/{dataset_name}_test.csv'

    plot_gan_transformer_results(df_path, model_name='GAN-Transformer', save=True)

import math
import json
import time
import sys
import argparse
import numpy as np
# from numpy.lib.twodim_base import triu_indices_from
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils.utils import mean_absolute_percentage_error as SMAPE
from utils.utils import plot_output, train_tf_enc_dec_gan, train_and_evaluate_tf_enc_dec_gan, test_model
from quaesita.pre_process_data import getSampledData
from quaesita.model import Transformer_EncoderDecoder_Seq2Seq, VanillaTransformer_seq2seq
from quaesita.TimeSeriesDataset import timeseriesDatasetCreateBatch 
# from utils.loss.dilate_loss import DILATE_loss
from quaesita.transformerGANs import VanillaTransformerGenerator, SequenceCritic
from utils.optimizer import MADGRAD, improved_gradient_penalty
from utils.utils import train_tf_encdec, test_tf_enc_dec
from utils.utils import root_mean_square_error as RMSE
from utils.utils import mean_absolute_scaled_error as MASE
import csv
import os
from sklearn.metrics import mean_absolute_error
from utils.utils import smape
from utils.utils import mae


torch.manual_seed(1)

EXP_FOLDER_PATH = 'cloudlabgpu1-output/TranImpWGAN/'

D_MODEL = int(sys.argv[1])
N_HEAD = int(sys.argv[2])
DROPOUT = float(sys.argv[3])

WINDOW_SIZE = sys.argv[4]
BATCH_SIZE = sys.argv[5]

DATASET_NAME = sys.argv[6]
GPU = sys.argv[7]
device = torch.device(GPU if torch.cuda.is_available() and GPU.startswith("cuda") else "cpu")

train_path = os.path.expanduser("~/wgan-gp-transformer/data/gan_train.csv")
val_path = os.path.expanduser("~/wgan-gp-transformer/data/gan_val.csv")
test_path = os.path.expanduser("~/wgan-gp-transformer/data/gan_test.csv")

def load_series_from_file(path):
    df = pd.read_csv(path)
    if 'CPU' in df.columns:
        return df['CPU'].values.astype(np.float32)
    else:
        raise ValueError(f"'CPU' column not found in {path}")

train_series = load_series_from_file(train_path)
val_series = load_series_from_file(val_path)
test_series = load_series_from_file(test_path)

print("✅ 데이터셋 로드 완료")
print("train_series:", train_series.shape)
print("val_series:", val_series.shape)
print("test_series:", test_series.shape)


# MinMax scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
train_series = scaler.fit_transform(train_series.reshape(-1, 1))
val_series = scaler.transform(val_series.reshape(-1, 1))
test_series = scaler.transform(test_series.reshape(-1, 1))

# FloatTensor 변환
train_series = torch.FloatTensor(train_series)
val_series = torch.FloatTensor(val_series)
test_series = torch.FloatTensor(test_series)


time_interval = 30
event_type = 0
scheduling_class = 1
layers_num = 6


#<-----------------------*-*-*-*------------------------>
# Training parameters 
#<-----------------------*-*-*-*------------------------>

WEIGHT_DECAY = 'DEFAULT' # weight decay for MADGRAD optimizer for Generator

lossG_type = 'mae'
criterionG = torch.nn.L1Loss()

# epochs 
epochs = 1

one = torch.ones([])
one = one.to(device)
mone = one * -1

print(one, mone)

#<-----------------------*-*-*-*------------------------>

def loss_quantile(mu:Variable, labels:Variable, quantile:Variable):
    loss = 0
    for i in range(mu.shape[1]):
        mu_e = mu[:, i].to(device)
        labels_e = labels[:, i].to(device)

        I = (labels_e >= mu_e).float().to(device)
        each_loss = 2*(torch.sum(quantile*((labels_e -mu_e)*I)+ (1-quantile) *(mu_e- labels_e)*(1-I))).to(device)
        loss += each_loss.to(device)

    return loss

def train(model, discriminator, criterionG, optimizer_G, optimizer_D,
          adverserial_loss, train_dl, dataset_params, scaler):
    
    model.train()
    batch_size = dataset_params['batch_size']
    forecasting_step = dataset_params['target_stride']
    window_size = dataset_params['window_size']

    x_input, truth, predicted = [], [], []
    train_loss, trainD_loss = 0, 0
    n = 0

    # ✅ 루프
    for x, y, tgt_mask in train_dl:
        x = x.to(device)
        y = y.to(device)

        optimizer_G.zero_grad()

        # ✅ Encoder-Decoder forward
        enc_out = model.encoder(model.positional_encoding(x))
        out = x[-1:].clone()
        outputs = x[-forecasting_step:].clone()

        for step in range(1, forecasting_step + 1):
            # ✅ decoder input 준비 (step마다 noise 추가)
            noise = torch.randn_like(out) * 0.01  # 이걸 dec_in에 추가
            dec_in = x[-1:] + noise if step == 1 else out + noise

            # ✅ mask 생성
            mask = torch.triu(torch.ones(1, 1)).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

            dec_in_emb = model.positional_encoding(dec_in)
            out = model.out(model.decoder(dec_in_emb, enc_out, mask))
            outputs[step - 1:step] = out

        y = y.unsqueeze(-1)

        # ✅ Discriminator input 준비
        def format_input(seq):  # [seq_len, B, 1]
            stacked = torch.stack([seq.squeeze(-1)[i] for i in range(seq.shape[0])], -1)
            return stacked.unsqueeze(-1)

        fake_input = format_input(torch.cat((x, outputs), 0)).to(device)
        real_input = format_input(torch.cat((x, y), 0)).to(device)

        # ✅ Discriminator 학습
        if adverserial_loss == 'improvedWGAN':
            # Enable grad for D
            for p in discriminator.parameters():
                p.requires_grad = True

            optimizer_D.zero_grad()
            d_real = discriminator(real_input).mean()
            d_fake = discriminator(fake_input.detach()).mean()

            # gradient penalty
            gp = improved_gradient_penalty(discriminator, real_input, fake_input.detach(), device)
            loss_d = d_fake - d_real + gp * 10  # ✅ 100 → 10으로 조정
            loss_d.backward()
            optimizer_D.step()
            trainD_loss += loss_d.item()

            # Disable grad for D
            for p in discriminator.parameters():
                p.requires_grad = False

            # ✅ Generator 학습
            g_d_fake = discriminator(fake_input).mean()
            alpha = 0.1
            loss_g = criterionG(outputs, y) + alpha * (-g_d_fake)
            loss_g.backward()
            optimizer_G.step()

            train_loss += loss_g.item() * x.shape[0]
            n += x.shape[0]

        # ✅ CPU로 복사 (detach 후 변환)
        x_np = x[0].detach().cpu().numpy().flatten().reshape(-1, 1)
        y_np = y[0].detach().cpu().numpy().flatten().reshape(-1, 1)
        pred_np = outputs[0].detach().cpu().numpy().flatten().reshape(-1, 1)

        x_input.append(scaler.inverse_transform(x_np))
        truth.append(scaler.inverse_transform(y_np))
        predicted.append(scaler.inverse_transform(pred_np))

    # ✅ MAE 계산
    avg_mae = mae(np.hstack([t.flatten() for t in truth]),
                  np.hstack([p.flatten() for p in predicted]))

    return train_loss / n, trainD_loss / n, x_input, truth, predicted, avg_mae

def train_generator(model,
            discriminator,
            criterionG,
            optimizer_G,
            train_dl,
            dataset_params,
            scaler):
    
    model.train()

    batch_size = dataset_params['batch_size']
    forecasting_step = dataset_params['target_stride']
    window_size = dataset_params['window_size']

    x_input = []
    truth = []
    predicted = []

    train_loss = 0
    trainD_loss = 0

    n = 0 

    for x, y, tgt_mask in train_dl:
        optimizer_G.zero_grad()
        x = x.to(device)
        y = y.to(device)

        # Adversarial ground truths
        valid = torch.autograd.Variable(torch.ones(window_size + 1, batch_size, 1, device=device), requires_grad=False)
        fake = torch.autograd.Variable(torch.zeros(window_size + 1, batch_size, 1, device=device), requires_grad=False)
        
        # encode x
        enc_out = model.encoder(model.positional_encoding(x)).to(device)

        out = torch.clone(x[-1:])   # last bit from the input
        outputs = torch.clone(x[-forecasting_step:])

        # decode x
        for step in range(1, forecasting_step + 1, 1):
            mask = (torch.triu(torch.ones(1, 1)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            tgt_mask = mask.to(device)

            # ✅ Teacher Forcing 적용 (50% 확률)
            if random.random() < 0.5 and step > 1:
                dec_in = y[step - 1:step]
            else:
                dec_in = out

            dec_in_emb = model.positional_encoding(dec_in).to(device)
            out = model.out(model.decoder(dec_in_emb, enc_out, tgt_mask))
            outputs[step - 1:step] = out
        
        y = y.unsqueeze(-1)

        # generator update
        loss = criterionG(outputs, y)
        loss.backward()
        optimizer_G.step()

        train_loss += (loss.item() * x.shape[0])
        n += x.shape[0]

        x = x.to('cpu') 
        y = y.to('cpu')
        outputs = outputs.detach().to('cpu')
        
        x_input.append(scaler.inverse_transform(np.reshape(np.array(x[0].view(-1).numpy()),(x.shape[1],1)))) # (x.shape[1],1))))
        truth.append(scaler.inverse_transform(np.reshape(np.array(y[0].view(-1).numpy()),(y.shape[1],1))))
        predicted.append(scaler.inverse_transform(np.reshape(np.array(outputs[0].view(-1).numpy()),(outputs.shape[1],1))))

        # Calculate MAE
    avg_mae = mae(np.hstack([t.flatten() for t in truth]), 
                  np.hstack([p.flatten() for p in predicted]))

    return train_loss/n, trainD_loss/n,  x_input, truth, predicted, avg_mae

def call_main(_window_size, _batch_size, _train_data, _cross_val_data, _test_data, _gpu_name):
    x_input, truth , predicted = [], [], []

    results = np.empty([0, 23], str)

    seq2seq_dataset_params = {
        'window_size': int(_window_size),
        'target_stride': 1,
        'batch_size': int(_batch_size),
        'flag': False
    }

    train_datasetA = timeseriesDatasetCreateBatch(_train_data, **seq2seq_dataset_params)
    test_datasetA = timeseriesDatasetCreateBatch(_test_data, **seq2seq_dataset_params)
    val_datasetA = timeseriesDatasetCreateBatch(_cross_val_data, **seq2seq_dataset_params)

    seq2seq_modelG_params = {
        'd_model': D_MODEL,
        'nhead': N_HEAD,
        'dropout': DROPOUT,
        'num_of_enc_layers': 1,
        'num_of_dec_layers': 1,
        'input_sequence_length': int(seq2seq_dataset_params['window_size']),
        'forecasting_step': 1
    }

    modelD_params = {
        'd_model': seq2seq_modelG_params['d_model'],
        'activation_fn': 'LeakyReLU'
    }

    modelG = Transformer_EncoderDecoder_Seq2Seq(**seq2seq_modelG_params).to(device)
    modelD = SequenceCritic(modelD_params).to(device)

    optimizer_G = MADGRAD(modelG.parameters(), lr=0.001)
    optimizer_D = MADGRAD(modelD.parameters(), lr=0.001)

    start_time = time.time()
    train_start_time = time.time() - start_time

    for e in tqdm(range(epochs)):
        trainG_loss, trainD_loss, x_input, truth, predicted, train_mae = train(
            model=modelG,
            discriminator=modelD,
            criterionG=criterionG,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            adverserial_loss='improvedWGAN',
            train_dl=train_datasetA,
            dataset_params=seq2seq_dataset_params,
            scaler=scaler
        )

    print(f"[Epoch {e+1}] G Loss: {trainG_loss:.6f}  D Loss: {trainD_loss:.6f}  Train MAE: {train_mae:.6f}")

    train_end_time = time.time() - start_time

    # ✅ 모델 저장
    save_path = f'RTX2080/best_saved_models/{DATASET_NAME}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(modelG.state_dict(), save_path)

    # ✅ 모델 로드 & eval 모드
    modelG.load_state_dict(torch.load(save_path))
    modelG.eval()

    def run_test_and_save(dataset, file_suffix):
        x_input, truth, predicted = test_tf_enc_dec(
            test_dl=dataset, model=modelG, scaler=scaler,
            forecasting_step=seq2seq_dataset_params['target_stride'], device=device
        )
        truth = np.hstack([t.flatten() for t in truth])
        predicted = np.hstack([p.flatten() for p in predicted])

        truth = scaler.inverse_transform(truth.reshape(-1, 1)).flatten()
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()

        mae_val = np.round(mae(truth, predicted), 4)
        rmse_val = np.round(RMSE(truth, predicted), 4)
        mase_val = MASE(truth, predicted)

        print(f"\n{file_suffix.upper()} MAE: {mae_val}  RMSE: {rmse_val}  MASE: {mase_val}")
        print(f"▶ {file_suffix.upper()} Predicted Mean: {np.mean(predicted):.6f} / Std: {np.std(predicted):.6f}")
        print(f"▶ {file_suffix.upper()} Actual Mean: {np.mean(truth):.6f} / Std: {np.std(truth):.6f}")


        # 저장
        df = pd.DataFrame({'truth': truth, 'predicted': predicted})
        df.to_csv(f'{output_dir}/{DATASET_NAME}_{file_suffix}.csv', index=False)

        return mae_val, rmse_val, mase_val

    output_dir = f'RTX2080/prediction-outputs/{DATASET_NAME}'
    os.makedirs(output_dir, exist_ok=True)

    train_inference_start_time = time.time() - start_time
    train_mae, train_rmse, train_mase = run_test_and_save(train_datasetA, 'train')
    train_inference_end_time = time.time() - start_time

    cv_inference_start_time = time.time() - start_time
    cv_mae, cv_rmse, cv_mase = run_test_and_save(val_datasetA, 'crossval')
    cv_inference_end_time = time.time() - start_time

    test_inference_start_time = time.time() - start_time
    test_mae, test_rmse, test_mase = run_test_and_save(test_datasetA, 'test')
    test_inference_end_time = time.time() - start_time

    # 정리
    training_time = train_end_time - train_start_time
    train_inference_time = train_inference_end_time - train_inference_start_time
    cv_inference_time = cv_inference_end_time - cv_inference_start_time
    test_inference_time = test_inference_end_time - test_inference_start_time

    results = np.append(results, [[
        DATASET_NAME, str(epochs), str(0.001), str(lossG_type),
        str(seq2seq_dataset_params['window_size']), str(seq2seq_dataset_params['batch_size']),
        str(seq2seq_modelG_params['dropout']), str(seq2seq_modelG_params['d_model']), str(seq2seq_modelG_params['nhead']),
        str(train_mae), str(train_rmse), str(train_mase),
        str(cv_mae), str(cv_rmse), str(cv_mase),
        str(test_mae), str(test_rmse), str(test_mase),
        str(_gpu_name), str(training_time),
        str(train_inference_time), str(cv_inference_time), str(test_inference_time)
    ]], axis=0)

    # 메모리 정리
    del modelG, modelD, optimizer_G, optimizer_D

    return results, truth, predicted


DATASET_NAME = sys.argv[6]

_train_data = pd.read_csv(train_path)["CPU"].values.astype("float32")
_cross_val_data = pd.read_csv(val_path)["CPU"].values.astype("float32")
_test_data = pd.read_csv(test_path)["CPU"].values.astype("float32")

scaler = MinMaxScaler(feature_range=(-1, 1))
_train_data = scaler.fit_transform(_train_data.reshape(-1, 1))
_cross_val_data = scaler.transform(_cross_val_data.reshape(-1, 1))
_test_data = scaler.transform(_test_data.reshape(-1, 1))

_train_data = torch.FloatTensor(_train_data)
_cross_val_data = torch.FloatTensor(_cross_val_data)
_test_data = torch.FloatTensor(_test_data)

test_len = len(_test_data)
output_dir = f'RTX2080/prediction-outputs/{DATASET_NAME}'
os.makedirs(output_dir, exist_ok=True)

lookback_set = np.round(np.arange(1,9,1) * 0.1 * test_len)         
print(lookback_set)
experiment_params = [[WINDOW_SIZE, BATCH_SIZE]]

# 실험 결과 저장을 위한 준비
results = np.empty([0,23], str)

for _window_size, _batch_size in experiment_params:
    _results, test_truth, test_predicted  = call_main(_window_size, _batch_size, _train_data, _cross_val_data, _test_data, GPU)
    results = np.append(results, _results, axis=0)

# 결과 저장을 위한 index 설정
index = [str(i) for i in range(1, len(results) + 1)]

# ✅ [1] 컬럼 정의 MAE 기준 23개)
columns = ['Dataset','Epoch','learning_rate','Cost function','Window Size','Batch Size','Dropout',
           'd_model','nhead','Train MAE','Train RMSE','Train MASE',
           'CV MAE','CV RMSE','CV MASE','Test MAE','Test RMSE','Test MASE',
           'GPU Name','training-time','train-inference-time','cv-inference-time','test-inference-time']

# ✅ [2] DataFrame 생성 (에러 방지용 try)
try:
    data_df = pd.DataFrame(results, index=index, columns=columns)
except Exception as e:
    print("❌ DataFrame 생성 실패:", e)
    print("results shape:", results.shape)
    print("예상 columns 개수:", len(columns))
    data_df = pd.DataFrame()  # 빈 DF 생성

# ✅ [3] 저장 경로 및 파일 설정
EXP_FOLDER_PATH = 'cloudlabgpu1-output/TranImpWGAN/'
os.makedirs(EXP_FOLDER_PATH, exist_ok=True)
csv_path = os.path.join(EXP_FOLDER_PATH, f"{DATASET_NAME}.csv")

# ✅ [4] 기존 CSV 파일 삭제 (MAPE 기반일 수 있으므로)
if os.path.exists(csv_path):
    print(f"⚠️ 기존 CSV(MAPE 기반)를 삭제하고 새로 만듭니다: {csv_path}")
    os.remove(csv_path)

# ✅ [5] 기존 파일이 여전히 존재할 경우 병합 (실제로는 삭제 후라 생략 가능)
if os.path.exists(csv_path):
    try:
        existing_file = pd.read_csv(csv_path)
        print("📂 기존 CSV 컬럼:", existing_file.columns.tolist())
        existing_file = existing_file[columns]  # SMAPE 컬럼만 유지
        data_df = pd.concat([data_df, existing_file], ignore_index=True)
    except Exception as e:
        print("⚠️ 기존 CSV 병합 실패, 건너뜀:", e)

# ✅ [6] 최종 저장
data_df.to_csv(csv_path, index=False)

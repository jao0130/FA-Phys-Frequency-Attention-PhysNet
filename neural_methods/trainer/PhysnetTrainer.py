"""PhysNet Trainer."""
import os
from collections import OrderedDict
import os
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.Lfre import FrequencyLoss
from neural_methods.model.My import PhysNet_NMF_My
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import csv
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm


class PhysnetTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        """Initializes the trainer with temporal autoencoder and PPG models."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.model_name = config.MODEL.NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.config = config
        self.epo =  config.PRETRAIN_EPOCH
        self.usepre = config.TRAIN.USE_PRETRAINED_WEIGHTS
        
        self.min_valid_loss = None
        self.best_epoch = 0
        self.task = config.TASK
        self.dataset_type = config.DATASET_TYPE
        self.train_state = config.TRAIN.DATA.INFO.STATE
        self.valid_state = config.VALID.DATA.INFO.STATE
        self.test_state = config.TEST.DATA.INFO.STATE
        self.lr = config.TRAIN.LR
        self.corr_lr = 5e-4
        self.gamma = 0.6
        
        self.ppg_model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.My.FRAME_NUM
        ).to(self.device)
        
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_pearson = Neg_Pearson()
            self.loss_mse = torch.nn.MSELoss()
            self.ppg_optimizer = optim.Adam(self.ppg_model.parameters(), lr=config.TRAIN.LR)
            self.ppg_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.ppg_optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, 
                steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("Invalid toolbox mode!")


    def train(self, data_loader):
        """Training routine with pretraining, partial freezing, and finetuning."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        
        accumulation_steps = 16 // self.batch_size
        mean_training_losses = []
        mean_valid_losses = []
        patience = 30
        counter = 0
        
        for epoch in range(self.max_epoch_num):
            print(f"\n==== Training Epoch: {epoch} ====")
            running_loss = 0.0
            running_loss_bvp = 0.0
            running_loss_spo2 = 0.0
            train_loss = []
            
            self.ppg_model.train()
            tbar = tqdm(data_loader["train"], ncols=120)
            
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Training epoch {epoch}")
                if self.dataset_type == "both":
                    face_data = batch[0].to(self.device)
                    finger_data = batch[1].to(self.device)
                    label_bvp = batch[2].to(self.device)
                    label_spo2 = batch[3].to(self.device).squeeze(-1)
                else:
                    face_data = batch[0].to(self.device)
                    finger_data = None
                    label_bvp = batch[1].to(self.device)
                    label_spo2 = batch[2].to(self.device).squeeze(-1)
                
                self.ppg_optimizer.zero_grad()
                pred_ppg = self.ppg_model(face_data)
                
                # BVP 損失
                loss_bvp = 0

                pred_ppg_norm = (pred_ppg - torch.mean(pred_ppg)) / (torch.std(pred_ppg) + 1e-8)
                label_bvp_norm = (label_bvp - torch.mean(label_bvp)) / (torch.std(label_bvp) + 1e-8)
                loss_pearson = self.loss_pearson(pred_ppg_norm, label_bvp_norm)


                loss_bvp =  loss_pearson
                total_loss = loss_bvp* 100

                # SpO2 損失
                
                total_loss = total_loss / accumulation_steps
                total_loss.backward()
                if (idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.ppg_model.parameters(), max_norm=1.0)
                    self.ppg_optimizer.step()
                    self.ppg_scheduler.step()             
                running_loss += total_loss.item() * accumulation_steps  # 還原損失
                running_loss_bvp += loss_bvp.item() * accumulation_steps if self.task in ["bvp", "both"] else 0
                train_loss.append(total_loss.item() * accumulation_steps)
                tbar.set_postfix(total_loss=total_loss.item() * accumulation_steps,loss_bvp =f"{running_loss_bvp/(idx+1):.4f}")
            
            mean_training_losses.append(np.mean(train_loss))
            print(f"train loss: {np.mean(train_loss)}")
            
            self.save_model(epoch, model_type="ppg", prefix="train")
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader, model_type="ppg")
                mean_valid_losses.append(valid_loss)
                with open ('D:\\SUMS-main\\SUMS-main\\rppg-Toolbox_SUMS\\data2\\lk\\loss.csv', 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    data_to_add = [
                        epoch+1, np.mean(train_loss), valid_loss
                    ]
                    csv_writer.writerow(data_to_add)
                print(f"Validation loss: {valid_loss}")
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    counter = 0
                    print(f"Update best ppg model! Best epoch: {self.best_epoch}")
                else:
                    counter += 1   
                if counter >= patience:
                    print(f"Early stopping at train epoch {epoch}")
                    break
            
        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"Best epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, [], self.config)

    def valid(self, data_loader, model_type="ppg"):
        """Runs the specified model on validation sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        
        print(f"\n==== Validating {model_type} Model ====")
        valid_loss = []
        sum_bvp = 0
        sum_fft_loss = 0
        model = self.ppg_model 
        model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=120)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                if self.dataset_type == "both":
                    face_data = valid_batch[0].to(self.device)
                    finger_data = valid_batch[1].to(self.device)
                    label_bvp = valid_batch[2].to(self.device)
                    label_spo2 = valid_batch[3].to(self.device).squeeze(-1)
                else:
                    face_data = valid_batch[0].to(self.device)
                    finger_data = None
                    label_bvp = valid_batch[1].to(self.device)
                    label_spo2 = valid_batch[2].to(self.device).squeeze(-1)
                
                pred_ppg = model(face_data)
                loss_bvp = 0
                loss_spo2 = 0                    

                pred_ppg_norm = (pred_ppg - torch.mean(pred_ppg)) / (torch.std(pred_ppg) + 1e-8)
                label_bvp_norm = (label_bvp - torch.mean(label_bvp)) / (torch.std(label_bvp) + 1e-8)
                loss_pearson = self.loss_pearson(pred_ppg_norm, label_bvp_norm)

                loss = loss_pearson* 100

                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item(),pearson = f"{sum_bvp/(valid_idx+1):.4f}",hr_mae = f"{sum_fft_loss/(valid_idx+1):.4f}")
            
        return np.mean(valid_loss)

    def test(self, data_loader):
        """Runs the PPG model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print("\n=== Testing PPG Model ===")
        rppg_predictions = dict()
        spo2_predictions = dict()
        rppg_labels = dict()
        spo2_labels = dict()
        
        header = [
            'V_TYPE', 'TASK', 'LR', 'Epoch Number', 'HR_MAE', 'HR_MAE_STD', 'HR_RMSE', 'HR_RMSE_STD',
            'HR_MAPE', 'HR_MAPE_STD', 'HR_Pearson', 'HR_Pearson_STD', 'HR_SNR', 'HR_SNR_STD',
            'SPO2_MAE', 'SPO2_MAE_STD', 'SPO2_RMSE', 'SPO2_RMSE_STD', 'SPO2_MAPE',
            'SPO2_MAPE_STD', 'SPO2_Pearson', 'SPO2_Pearson_STD', 'SPO2_SNR', 'SPO2_SNR_STD',
            'Model', 'train_state', 'valid_state', 'test_state'
        ]
        
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.ppg_model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, f"{self.model_name}_{self.model_file_name}_Epoch_{str(self.max_epoch_num - 1)}.pth")
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.ppg_model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, f"{self.model_name}_{self.model_file_name}_Epoch_{str(self.best_epoch)}.pth")
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.ppg_model.load_state_dict(torch.load(best_model_path))

        self.ppg_model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=120)):
                if self.dataset_type == "both":
                    face_data = test_batch[0].to(self.device)
                    finger_data = test_batch[1].to(self.device)
                    rppg_label = test_batch[2].to(self.device)
                    spo2_label = test_batch[3].to(self.device).squeeze(-1)
                    subj_index = test_batch[4]
                    sort_index = test_batch[5]
                else:
                    face_data = test_batch[0].to(self.device)
                    finger_data = None
                    rppg_label = test_batch[1].to(self.device)
                    spo2_label = test_batch[2].to(self.device).squeeze(-1)
                    subj_index = test_batch[3]
                    sort_index = test_batch[4]
                
                pred_ppg = self.ppg_model(face_data)
                pred_ppg = pred_ppg.cpu()
                rppg_label = rppg_label.cpu()
                
                batch_size = test_batch[0].shape[0]
                for idx in range(batch_size):
                    subj = subj_index[idx]
                    sort = int(sort_index[idx].replace('_', ''))
                    if subj not in rppg_predictions:
                        rppg_predictions[subj] = dict()
                        rppg_labels[subj] = dict()
                    rppg_predictions[subj][sort] = pred_ppg[idx]
                    rppg_labels[subj][sort] = rppg_label[idx]
        
        file_exists = os.path.isfile('D:\\SUMS-main\\SUMS-main\\rppg-Toolbox_SUMS\\data2\\lk\\result2.csv')
        with open('D:\\SUMS-main\\SUMS-main\\rppg-Toolbox_SUMS\\data2\\lk\\result2.csv', 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                csv_writer.writerow(header)
            
            epoch_num = self.best_epoch
            if self.task == "bvp":
                result = calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
                metrics = result["metrics"]
                HR_MAE, HR_MAE_STD = metrics.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics.get("FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics.get("FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics.get("FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics.get("FFT_SNR", (None, None)) if "FFT_SNR" in metrics else (None, None)
                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num, HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    "/", "/", "/", "/", "/", "/", "/", "/", "/", "/",
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]
            elif self.task == "spo2":
                result = calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
                metrics = result["metrics"]
                SPO2_MAE, SPO2_MAE_STD = metrics.get("FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics.get("FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics.get("FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics.get("FFT_Pearson", (None, None))
                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num, "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/",
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/",
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]
            elif self.task == "both":
                result_rppg = calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
                result_spo2 = calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
                metrics_rppg = result_rppg["metrics"]
                HR_MAE, HR_MAE_STD = metrics_rppg.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics_rppg.get("FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics_rppg.get("FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics_rppg.get("FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics_rppg.get("FFT_SNR", (None, None))
                metrics_spo2 = result_spo2["metrics"]
                SPO2_MAE, SPO2_MAE_STD = metrics_spo2.get("FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics_spo2.get("FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics_spo2.get("FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics_spo2.get("FFT_Pearson", (None, None))
                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num, HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/",
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]
                
                if self.config.INFERENCE.MODEL_PATH:
                    epoch_number = self.extract_epoch_from_path(self.config.INFERENCE.MODEL_PATH)
                    data_to_add_hr_spo2_MAE = [epoch_number, HR_MAE, SPO2_MAE]
            
            if self.config.TOOLBOX_MODE != "only_test":
                csv_writer.writerow(data_to_add)
            else:
                with open("D:\\SUMS-main\\SUMS-main\\rppg-Toolbox_SUMS\\data2\\lk\\MAE.csv", 'a', newline='') as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(data_to_add_hr_spo2_MAE)
                
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(rppg_predictions, rppg_labels, self.config, data_type='rppg')
            self.save_test_outputs(spo2_predictions, spo2_labels, self.config, data_type='spo2')

    def save_model(self, index, model_type="ppg", prefix=""):
        """Saves the specified model's state dict."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, f"{self.model_name}_{self.model_file_name}_Epoch_{index}.pth")
        torch.save(self.ppg_model.state_dict(), model_path)
        print(f"Saved {model_type} Model Path: {model_path}")

    def save_test_outputs(self, predictions, labels, config, data_type=''):
        """Saves test outputs to a .npz file."""
        if not os.path.exists(config.TEST.OUTPUT_SAVE_DIR):
            os.makedirs(config.TEST.OUTPUT_SAVE_DIR)
        output_file = os.path.join(config.TEST.OUTPUT_SAVE_DIR, f"{self.model_file_name}_{data_type}_test_outputs.npz")
        np.savez(output_file, predictions=predictions, labels=labels)
        print(f"Saved test outputs to: {output_file}")

    def extract_epoch_from_path(self, model_path):
        """Extracts epoch number from model path."""
        parts = model_path.split('/')
        for part in parts:
            if 'Epoch' in part:
                a = part.find("Epoch")
                b = part.find(".pth")
                epoch_str = part[a+5:b]
                return int(epoch_str)+1
        raise ValueError("The model path does not contain an epoch number.")
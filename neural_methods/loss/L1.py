# neural_methods/loss/FrequencyLoss.py
import torch
import torch.nn as nn
import torch.fft
import numpy as np

class L1(nn.Module):
    def __init__(self, fs=30, hr_range=(40, 180), num_bins=140):
        super(L1, self).__init__()
        self.fs = fs  # 採樣率 (Hz)
        self.hr_range = hr_range  # 心率範圍 (bpm)
        self.num_bins = num_bins  # 頻率 bin 數量
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, spr, hr_gt):
        batch, length = spr.shape

        # 檢查輸入
        assert torch.all((hr_gt >= self.hr_range[0]) & (hr_gt <= self.hr_range[1])), \
            f"hr_gt out of range [{self.hr_range[0]}, {self.hr_range[1]}]: {hr_gt}"

        # 計算 PSD
        spr = spr - torch.mean(spr, dim=1, keepdim=True)
        fft = torch.fft.rfft(spr, dim=1)
        psd = torch.abs(fft) ** 2

        # 頻率軸
        freqs = torch.linspace(0, self.fs / 2, fft.shape[1], device=spr.device) * 60
        hr_min, hr_max = self.hr_range
        mask = (freqs >= hr_min) & (freqs <= hr_max)
        psd = psd[:, mask]
        freqs = freqs[mask]

        # 插值到 num_bins
        target_freqs = torch.linspace(hr_min, hr_max, self.num_bins, device=spr.device)
        psd_interpolated = torch.zeros(batch, self.num_bins, device=spr.device)
        freqs_np = freqs.cpu().detach().numpy()
        target_freqs_np = target_freqs.cpu().detach().numpy()
        for i in range(batch):
            psd_np = psd[i].cpu().detach().numpy()
            psd_interpolated[i] = torch.from_numpy(
                np.interp(target_freqs_np, freqs_np, psd_np)
            ).to(spr.device)

        # 正規化 PSD
        psd_sum = torch.sum(psd_interpolated, dim=1, keepdim=True)
        psd_interpolated = psd_interpolated / (psd_sum + 1e-8)

        # 計算預測心率（加權平均）
        pred_hr = torch.sum(psd_interpolated * target_freqs, dim=1)

        # 使用平滑 L1 損失
        loss = torch.nn.functional.smooth_l1_loss(pred_hr, hr_gt)

        return loss
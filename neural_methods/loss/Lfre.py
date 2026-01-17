import torch
import torch.nn as nn
import torch.fft
import numpy as np

class FrequencyLoss(nn.Module):
    def __init__(self, fs=30, hr_range=(40, 180), num_bins=140):
        super(FrequencyLoss, self).__init__()
        self.fs = fs  # 採樣率 (Hz)
        self.hr_range = hr_range  # 心率範圍 (bpm)
        self.num_bins = num_bins  # 頻率 bin 數量
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, spr, hr_gt):
        """
        spr: 預測的 rPPG 訊號，形狀 (batch, length)
        hr_gt: ground-truth 心率 (bpm)，形狀 (batch,) 或 (batch, length)
        """
        batch, length = spr.shape

        # 處理 hr_gt 的形狀
        if hr_gt.dim() == 2:  # 如果 hr_gt 是 (batch, length)
            hr_gt = torch.median(hr_gt, dim=1).values  # 取中位數，轉為 (batch,)
        elif hr_gt.dim() != 1:  # 確保 hr_gt 是 (batch,)
            raise ValueError(f"hr_gt 應為形狀 (batch,) 或 (batch, length)，但得到 {hr_gt.shape}")

        # 檢查 hr_gt 是否在範圍內
        assert torch.all((hr_gt >= self.hr_range[0]) & (hr_gt <= self.hr_range[1])), \
            f"hr_gt 超出範圍 [{self.hr_range[0]}, {self.hr_range[1]}]: {hr_gt}"

        # 計算 PSD
        spr = spr - torch.mean(spr, dim=1, keepdim=True)  # 去均值
        fft = torch.fft.rfft(spr, dim=1)  # 實數 FFT
        psd = torch.abs(fft) ** 2  # 功率譜密度

        # 頻率軸
        freqs = torch.linspace(0, self.fs / 2, fft.shape[1], device=spr.device) * 60  # 轉換為 bpm
        hr_min, hr_max = self.hr_range
        mask = (freqs >= hr_min) & (freqs <= hr_max)
        psd = psd[:, mask]
        freqs = freqs[mask]

        # 將 PSD 插值到 num_bins
        target_freqs = torch.linspace(hr_min, hr_max, self.num_bins, device=spr.device)
        psd_interpolated = torch.zeros(batch, self.num_bins, device=spr.device)

        # 使用 np.interp 進行插值
        freqs_np = freqs.cpu().detach().numpy()  # 轉為 NumPy 陣列，移除梯度
        target_freqs_np = target_freqs.cpu().detach().numpy()
        for i in range(batch):
            psd_np = psd[i].cpu().detach().numpy()  # 移除梯度
            psd_interpolated[i] = torch.from_numpy(
                np.interp(target_freqs_np, freqs_np, psd_np)
            ).to(spr.device)

        # 正規化 PSD
        psd_sum = torch.sum(psd_interpolated, dim=1, keepdim=True)
        psd_interpolated = psd_interpolated / (psd_sum + 1e-8)  # 避免除零

        # 檢查 PSD 是否包含 NaN
        assert not torch.any(torch.isnan(psd_interpolated)), "NaN in psd_interpolated"

        # 將 ground-truth HR 轉為類別索引
        hr_bins = torch.linspace(hr_min, hr_max, self.num_bins, device=spr.device)
        hr_gt_idx = torch.argmin(torch.abs(hr_gt.unsqueeze(1) - hr_bins), dim=1)

        # 計算交叉熵損失
        loss = self.ce_loss(psd_interpolated, hr_gt_idx)

        return loss
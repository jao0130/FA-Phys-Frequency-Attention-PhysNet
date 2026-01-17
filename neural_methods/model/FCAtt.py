import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalShift(nn.Module):
    def __init__(self, shift_ratio=0.125):
        super(TemporalShift, self).__init__()
        self.shift_ratio = shift_ratio

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.size()
        shift = int(C * self.shift_ratio)
        if shift == 0:
            return x
        # Shift channels along time dimension
        x_shift = x.clone()
        x_shift[:, :shift, :-1] = x[:, :shift, 1:]  # Forward shift
        x_shift[:, shift:2*shift, 1:] = x[:, shift:2*shift, :-1]  # Backward shift
        x_shift[:, 2*shift:] = x[:, 2*shift:]  # No shift
        return x_shift
        
class FrequencyAttention(nn.Module):
    def __init__(self, num_input_channels, frames=128, fps=30, pool_size = 18, temporal_kernel_size=3, M_intermediate_channels=16): # 新增 M_intermediate_channels
        super(FrequencyAttention, self).__init__()
        self.frames = frames
        self.fps = fps
        self.pool_size_h = pool_size
        self.pool_size_w = pool_size
        self.pool = nn.AdaptiveAvgPool3d((None, self.pool_size_h, self.pool_size_w))

        # 可學習的注意力分數精煉器 (Learnable Score Refiner)
        self.score_refiner = nn.Sequential(
            # 第一層：從多個輸入通道中學習共通的時序模式
            nn.Conv3d(
                in_channels=num_input_channels,     # 例如 64
                out_channels=M_intermediate_channels, # 例如 16 或 32
                kernel_size=(temporal_kernel_size, 1, 1), # 時序卷積
                padding=((temporal_kernel_size - 1) // 2, 0, 0),
                groups=1,  # <--- 關鍵：濾波器共享於所有輸入通道，學習共通模式
                bias=False
            ),
            nn.BatchNorm3d(M_intermediate_channels),
            nn.ELU(inplace=True),

            # 第二層：將共通模式特徵整合為單一的信心分數通道
            nn.Conv3d(
                in_channels=M_intermediate_channels,
                out_channels=1, # <--- 關鍵：輸出單一通道的分數圖
                kernel_size=(1, 1, 1), # 1x1x1 卷積進行特徵整合與降維
                bias=True # 最終分數可以有偏置
            )
        )

        self.spatial_score_refiner = nn.Sequential(
            nn.Conv2d(num_input_channels, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),  # 在多通道時用 BN
            nn.ELU(inplace=True),
   
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T, H, W = x.size() # C 就是 num_input_channels

        # 1. 空間池化和重塑
        x_pooled = self.pool(x)
        x_reshaped = x_pooled.view(B, C, T, -1)

        # 2. 傅立葉變換
        fft_out = torch.fft.rfft(x_reshaped, dim=2, norm='ortho')

        # 3. 頻率遮罩
        freqs = torch.fft.rfftfreq(T, d=1./self.fps).to(x.device)
        mask = (freqs >= 0.7) & (freqs <= 2.5)
        fft_filtered_complex = fft_out * mask.view(1, 1, -1, 1)

        filtered_wave = torch.fft.irfft(fft_filtered_complex, n=T, dim=2, norm='ortho')
        filtered_wave_spatial = filtered_wave.view(B, C, T, self.pool_size_h, self.pool_size_w)
        temporal_scores = self.score_refiner(filtered_wave_spatial)
        pattern_map_scores = torch.mean(temporal_scores, dim=2)
        
        # A-4. 使用 Sigmoid 得到「模式正確性」的機率圖 (0-1之間)
        pattern_map_probs = torch.sigmoid(pattern_map_scores)

        power_map = torch.mean(torch.abs(fft_filtered_complex) ** 2, dim=2)  # [B, C, h*w]
        spatial_att = power_map.view(B, C, self.pool_size_h, self.pool_size_w)  # [B, C, h, w]
        spatial_score = self.spatial_score_refiner(spatial_att)
        energy_map_probs = torch.sigmoid(spatial_score)

        att_map_fused = torch.sqrt(pattern_map_probs * energy_map_probs + 1e-20)
        
        att_map_upsampled = F.interpolate(
        att_map_fused,
        size=(H, W),
        mode='bilinear',
        align_corners=False
        )        
        # 9. 將「共享的」注意力圖應用於原始輸入 x 的所有通道
        att_map_upsampled = att_map_upsampled.unsqueeze(2)

        output = x * att_map_upsampled

        return output

    
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv3d(2, 32, kernel_size=[3, 1, 1], stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=[3, 1, 1], stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)


    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.conv3(x)
        return x

class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128, drop_rate1=0.1, drop_rate2=0.2):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ELU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1, 3, 3], stride=1, padding=[0, 1, 1]),  # Replace MaxpoolSpa
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),  # Replace MaxpoolSpa
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [5, 1, 1], stride=1, padding=[2, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )

        self.residual_conv2 = nn.Conv3d(16, 32, 1, stride=1, padding=0)
        self.residual_conv4 = nn.Conv3d(64, 64, 1, stride=1, padding=0)
        self.residual_conv6 = nn.Conv3d(64, 64, 1, stride=1, padding=0)
        self.residual_conv8 = nn.Conv3d(64, 64, 1, stride=1, padding=0)

        self.temporal_shift = TemporalShift(shift_ratio=0.125)

        self.channel_attention = ChannelAttention3D(in_channels=64, reduction=4)
        self.channel_attention1 = ChannelAttention3D(in_channels=64, reduction=4)
        self.channel_attention2 = ChannelAttention3D(in_channels=64, reduction=4)
        self.channel_attention3 = ChannelAttention3D(in_channels=64, reduction=4)

        self.drop_1 = nn.Dropout3d(drop_rate1)
        self.drop_2 = nn.Dropout3d(drop_rate1)
        self.drop_3 = nn.Dropout3d(drop_rate2)
        self.drop_4 = nn.Dropout3d(drop_rate2)
        self.drop_5 = nn.Dropout3d(drop_rate2)
        self.drop_6 = nn.Dropout3d(drop_rate2)
        self.drop_7 = nn.Dropout3d(drop_rate1)
        self.drop_8 = nn.Dropout3d(drop_rate1)
        self.drop_9 = nn.Dropout3d(drop_rate1)

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        #self.att_conv1 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)
        self.att_mask1 = FrequencyAttention(
                num_input_channels=64, 
                frames=frames,
                fps=30, 
                pool_size=18, # 假設 pool_size 固定或作為參數傳入
                temporal_kernel_size=7
            )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        #--- SpO2 分支 ---
        self.spo2_branch_conv1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=[1,3,3], padding=[0,1,1]), # 空間特徵提取，通道降維
            nn.BatchNorm3d(32), nn.ELU(inplace=True))
        self.spo2_branch_conv2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=[3,1,1], padding=[1,0,0]), # 時間特徵聚合，通道再降維
            nn.BatchNorm3d(16), nn.ELU(inplace=True))
        self.spo2_branch_pool = nn.AdaptiveAvgPool3d((frames, 1, 1)) # 時間對齊到 self.frames

        # --- SpO2 預測頭 (融合原始rPPG, VPG, APG 和 SpO2分支特徵) ---
        self.num_spo2_branch_output_channels = 16 
        # 融合頭輸入通道 = 1 (rPPG) + 1 (VPG) + 1 (APG) + 16 (SpO2分支) = 19
        fusion_head_input_channels = 1 + 1 + 1 + self.num_spo2_branch_output_channels
        self.spo2_fusion_head = nn.Sequential(
            nn.Conv1d(fusion_head_input_channels, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm1d(32),
            nn.ELU(),

            ChannelAttention1D(in_channels=32, reduction=8),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1), nn.ELU(),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
            )

        self.fusion_net = FusionNet()

    def forward(self, x1, x2=None):
        [batch, channel, length, width, height] = x1.shape
        if x2 is not None:
            rppg_wave1, spo2 = self.encode_video(x1)
            rppg_wave2, _ = self.encode_video(x2)
            x = self.fusion_net(rppg_wave1, rppg_wave2)
        else:
            rppg_wave, spo2 = self.encode_video(x1)

        rppg_wave = rppg_wave.view(batch, length) # [B, self.frames]

        rppg_d1 = rppg_wave.unsqueeze(1)
        vpg_temp = torch.diff(rppg_d1, n=1, dim=-1)
        vpg_signal = F.pad(vpg_temp, (0, 1), mode='replicate')
        apg_temp = torch.diff(vpg_signal, n=1, dim=-1) 
        apg_signal = F.pad(apg_temp, (0, 1), mode='replicate')

        rppg_for_fusion = rppg_d1 # [B, 1, self.frames]
        spo2_branch_for_fusion = spo2.view(
            batch, self.num_spo2_branch_output_channels, length
        )
        fused_features_1d = torch.cat(
            (rppg_for_fusion, vpg_signal, apg_signal, spo2_branch_for_fusion),
            dim=1
        )
        spo2_pred = self.spo2_fusion_head(fused_features_1d) # [B, 1]
        spo2_pred = spo2_pred * 15.0 + 85.0

        return rppg_wave, spo2_pred

    def encode_video(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        #x = self.temporal_shift(x)
        x = self.ConvBlock3(x)
        x_att = self.att_mask1(x) 
        #x_ca = self.channel_attention(x_att) 
        x = self.MaxpoolSpaTem(x_att) 
        #x = self.drop_1(x)

        residual4 = self.residual_conv4(x)
        x = self.ConvBlock4(x)
        #x_ca1 = self.channel_attention1(x)
        x = x + residual4
        x = self.ConvBlock5(x)
        #x = self.drop_3(x)
        
        residual6 = self.residual_conv6(x)
        x = self.ConvBlock6(x)
        #x_ca2= self.channel_attention2(x)  
        x = x + residual6
        xb = self.ConvBlock7(x)
        #x = self.drop_4(x)

        # --- HR 分支 ---
        residual8 = self.residual_conv8(xb)
        x = self.ConvBlock8(xb)
        #x_ca3= self.channel_attention3(x)        
        x = x + residual8
        x = self.ConvBlock9(x)
        #x = self.drop_5(x)

        hr_f = self.upsample(x)
        hr_f = self.upsample2(hr_f)
        hr_f_pooled = self.poolspa(hr_f)
        #x = self.drop_7(x)
        rppg_wave_features = self.ConvBlock10(hr_f_pooled)

        # --- SpO2 分支 ---
        spo2_b_f = self.spo2_branch_conv1(xb)
        spo2_b_f = self.spo2_branch_conv2(spo2_b_f)
        spo2_branch_features = self.spo2_branch_pool(spo2_b_f)

        return rppg_wave_features, spo2_branch_features
    
class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=8): 
        super().__init__()
        if in_channels < reduction: 
            reduction = in_channels // 2 if in_channels // 2 > 0 else in_channels

        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): 
        y = self.avg_pool(x) 
        y = self.fc(y)      
        return x * y         

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention
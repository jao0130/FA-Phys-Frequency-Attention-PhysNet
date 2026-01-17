import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics.pairwise import cosine_similarity
import torch

def compute_self_correlation(signal, fs, lowcut=0.7, highcut=3.0, window_size=30, step=2):
    signal = np.asarray(signal).flatten()
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    signal_len = len(filtered_signal)
    num_windows = (signal_len - window_size) // step + 1
    windows = [filtered_signal[i*step : i*step + window_size] for i in range(num_windows)]
    
    windows_array = np.array(windows)
    corr_matrix = cosine_similarity(windows_array, windows_array)
    
    # 標準化到 [0, 1]
    corr_matrix = (corr_matrix - corr_matrix.min()) / (corr_matrix.max() - corr_matrix.min())
    
    corr_matrix = torch.from_numpy(corr_matrix).float()
    return corr_matrix
"""The dataloader for UBFC datasets.

Details for the UBFC-RPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd
from evaluation.post_process import _calculate_fft_hr



class UBFCLoader(BaseLoader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an UBFC-PHYS dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- s1/
                     |       |-- vid_s1_T1.avi
                     |       |-- vid_s1_T2.avi
                     |       |-- vid_s1_T3.avi
                     |       |...
                     |       |-- bvp_s1_T1.csv
                     |       |-- bvp_s1_T2.csv
                     |       |-- bvp_s1_T3.csv
                     |   |-- s2/
                     |       |-- vid_s2_T1.avi
                     |       |-- vid_s2_T2.avi
                     |       |-- vid_s2_T3.avi
                     |       |...
                     |       |-- bvp_s2_T1.csv
                     |       |-- bvp_s2_T2.csv
                     |       |-- bvp_s2_T3.csv
                     |...
                     |   |-- sn/
                     |       |-- vid_sn_T1.avi
                     |       |-- vid_sn_T2.avi
                     |       |-- vid_sn_T3.avi
                     |       |...
                     |       |-- bvp_sn_T1.csv
                     |       |-- bvp_sn_T2.csv
                     |       |-- bvp_sn_T3.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-PHYS dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "s*" + os.sep + "*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'vid_(.*).avi', data_dir).group(1), "path": data_dir} for data_dir in data_dirs]
        return dirs
    
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        video_file = data_dirs[i]['path']
        frames = self.get_frame_count(video_file)        

        # 讀取 BVP 數據
        bvps = self.read_wave(
            os.path.join(os.path.dirname(data_dirs[i]['path']),"bvp_{0}.csv".format(saved_filename)))

        bvps = BaseLoader.resample_ppg(bvps, frames)
        resampled_bvp = self._bandpass_filter(
            bvps, fs=self.config_data.FS, lowcut=0.5, highcut=4.0
        )
        bvps = self._detrend(resampled_bvp)
        # 初始化儲存列表
        all_input_name_list = []
        chunk_idx = 0
        
        # 迭代 read_video 的生成器
        
        for frames_chunk in self.read_video(video_file, 320):

            start_frame = chunk_idx * 320
            end_frame = start_frame + frames_chunk.shape[0]
            bvps_chunk = bvps[start_frame:end_frame] if bvps is not None else None
            frames_clips, bvps_clips = self.preprocess(
                frames_chunk, bvps_chunk, config_preprocess, video_type="face"
            )
            input_name_list, label_name_list = self.save_multi_process(
                frames_clips, bvps_clips, saved_filename, chunk_idx
            )
            all_input_name_list.extend(input_name_list)
            
            chunk_idx += 1
        
        file_list_dict[i] = all_input_name_list
    
    def preprocess(self, frames, bvps, config_preprocess, video_type):
        """Preprocesses a pair of data."""

        frames = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H
        )        

        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)

        if bvps is not None :
            if config_preprocess.LABEL_TYPE == "Raw":
                pass
            elif config_preprocess.LABEL_TYPE == "DiffNormalized":
                bvps = BaseLoader.diff_normalize_label(bvps)
            elif config_preprocess.LABEL_TYPE == "Standardized":
                bvps = BaseLoader.standardized_label(bvps)
            else:
                raise ValueError("Unsupported label type!")

            if config_preprocess.DO_CHUNK:
                frames_clips, bvps_clips = self.chunk_only_bvp(
                    data, bvps, config_preprocess.CHUNK_LENGTH, video_type
                    )
            else:
                frames_clips = np.array([data])
                bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips
    
    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            if self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST :
                # Skip loading the input as it's in the exclusion list
                continue
            if self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST):
                # Skip loading the input as it's not in the task list
                continue
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("face_input", "hr") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)


    def preprocess_dataset_subprocess_raw(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        frames = self.read_video(
            os.path.join(data_dirs[i]['path'],"vid.avi"))
        bvps = self.read_wave(
            os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list


    @staticmethod
    def read_video_raw(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)
    
    @staticmethod
    def read_video(video_file, chunk_size=320):
        """Reads a video file in chunks using a generator to minimize memory usage."""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        target_size = (512, 512)
        chunk = []
        total_frames = 0
        chunk_idx = 0

        while True:
            success, frame = VidObj.read()
            if not success:
                if chunk:
                    yield np.array(chunk, dtype=np.uint8)
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            chunk.append(resized_frame)
            total_frames += 1

            if len(chunk) == chunk_size:
                yield np.array(chunk, dtype=np.uint8)
                chunk = []
                chunk_idx += 1

        VidObj.release()

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp = []
        with open(bvp_file, "r") as f:
            d = csv.reader(f)
            for row in d:
                bvp.append(float(row[0]))
        return np.asarray(bvp)
    
    @staticmethod
    def get_frame_count(video_file):
        """Detects the total number of frames in a video file efficiently."""
        import cv2
        cap = cv2.VideoCapture(video_file)
        # 獲取幀數
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        cap.release()
        return frame_count
    
    def save_multi_process(self, frames_clips, bvps_clips, filename, clip_idx):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volume pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            spo2_clips(np.array): SpO2 values for each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        
        count = 0 + 3 *clip_idx
        input_path_name_list = []
        label_path_name_list = []


        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels_bvp))
            assert (len(self.inputs) == len(self.labels_spo2))
            input_path_name = os.path.join(self.cached_path, f"{filename}_face_input_{count}.npy")
            label_path_name = os.path.join(self.cached_path, f"{filename}_hr_{count}.npy")


            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)


            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])


            count += 1
        
        return input_path_name_list, label_path_name_list

    def __getitem__(self, index):
        """Returns a clip of video and its corresponding signals."""
        label_bvp = np.load(self.labels[index])
        spo2 = np.zeros(label_bvp.shape[0], dtype=np.float32) 
        label_bvp = np.float32(label_bvp)

        data = np.load(self.inputs[index])
        data = np.transpose(data, (3, 0, 1, 2))
        data = np.float32(data)
        item_path = self.inputs[index]

        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.rindex('_input_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + len('_input_'):].split('.')[0]

        return data, label_bvp, spo2 ,filename, chunk_id
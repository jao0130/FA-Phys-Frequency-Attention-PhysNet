"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import glob
import json
import os
import re
import csv
from scipy.interpolate import interp1d

import cv2
import pandas as pd
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
from evaluation.post_process import _calculate_fft_hr

class PURELoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 01-01/
                     |      |-- 01-01/
                     |      |-- 01-01.json
                     |   |-- 01-02/
                     |      |-- 01-02/
                     |      |-- 01-02.json
                     |...
                     |   |-- ii-jj/
                     |      |-- ii-jj/
                     |      |-- ii-jj.json
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = int(subject_trail_val)
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an empty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        
        frames = self.read_video(
            os.path.join(data_dirs[i]['path'], filename, ""))
        files = os.path.join(data_dirs[i]['path'], "{0}.json".format(filename))
        bvps = self.read_wave(files) 
        spo2 = self.read_spo2(files) 
        target_length = frames.shape[0]  
   
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        spo2 = BaseLoader.resample_ppg(spo2, target_length)

        #bvps = self._bandpass_filter(
        #    bvps, fs=self.config_data.FS, lowcut=0.5, highcut=3.5
        #)
        
        frames_clips, bvps_clips, spo2_clips = self.preprocess(frames, bvps, spo2, config_preprocess)

        input_name_list, label_name_list, spo2_name_list = self.save_multi_process(frames_clips, bvps_clips, spo2_clips, saved_filename,0)
        file_list_dict[i] = input_name_list

    def preprocess(self, frames, bvps, spo2, config_preprocess):
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
            config_preprocess.RESIZE.H)
        
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

        if bvps is not None and spo2 is not None:
            if config_preprocess.LABEL_TYPE == "Raw":
                pass
            elif config_preprocess.LABEL_TYPE == "DiffNormalized":
                bvps = BaseLoader.diff_normalize_label(bvps)
            elif config_preprocess.LABEL_TYPE == "Standardized":
                bvps = BaseLoader.standardized_label(bvps)
            else:
                raise ValueError("Unsupported label type!")

            if config_preprocess.DO_CHUNK:
                frames_clips, bvps_clips, spo2_clips = self.chunk(data, bvps, spo2, config_preprocess.CHUNK_LENGTH, 'face')
            else:
                frames_clips = np.array([data])
                bvps_clips = np.array([bvps])
                spo2_clips = np.array([spo2])


            return frames_clips, bvps_clips, spo2_clips
        else:
            if config_preprocess.DO_CHUNK:
                frames_clips, _, _ = self.chunk(data, None, None, config_preprocess.CHUNK_LENGTH,'finger')
            else:
                frames_clips = np.array([data])

            return frames_clips, None, None
        
    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()

        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        inputs_only_face = [input_file for input_file in inputs if "finger_input" not in input_file]
        labels_bvp = [input_file.replace("face_input", "hr") for input_file in inputs_only_face]
        labels_spo2 = [input_file.replace("face_input", "spo2") for input_file in inputs_only_face]

        self.inputs = inputs
        self.labels_bvp = labels_bvp
        self.labels_spo2 = labels_spo2


        self.preprocessed_data_len = len(inputs)
    
    
    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
    
    @staticmethod
    def read_spo2(spo2_file):
        """Reads a SpO2 signal file and corresponding timestamps."""
        with open(spo2_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["o2saturation"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)

    
    def __getitem__(self, index):
        """Returns a clip of video and its corresponding signals."""
        label_bvp = np.load(self.labels_bvp[index])
        label_spo2 = np.load(self.labels_spo2[index])

        label_bvp = np.float32(label_bvp)
        label_spo2 = np.float32(label_spo2)

        if label_spo2.ndim == 1:
            label_spo2 = label_spo2.reshape(1, -1).mean(axis=1, keepdims=True)

        if self.dataset_type == "face":
            data = np.load(self.inputs[index])
            data = np.transpose(data, (3, 0, 1, 2))
            data = np.float32(data)
            item_path = self.inputs[index]
        elif self.dataset_type == "finger":
            data = np.load(self.inputs_finger[index])
            data = np.transpose(data, (3, 0, 1, 2))
            data = np.float32(data)
            item_path = self.inputs_finger[index]
        elif self.dataset_type == "both":
            data_face = np.load(self.inputs[index])
            data_finger = np.load(self.inputs_finger[index])
            data_face = np.transpose(data_face, (3, 0, 1, 2))
            data_finger = np.transpose(data_finger, (3, 0, 1, 2))
            data_face = np.float32(data_face)
            data_finger = np.float32(data_finger)
            item_path = self.inputs[index]
            item_path_2 = self.inputs_finger[index]
            item_path_filename_2 = item_path_2.split(os.sep)[-1]

        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.rindex('_input_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + len('_input_'):].split('.')[0]

        if self.dataset_type == "both":
            return data_face, data_finger, label_bvp, label_spo2, filename, chunk_id
        else:
            return data, label_bvp, label_spo2, filename, chunk_id
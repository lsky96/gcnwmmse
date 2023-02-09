"""
Author: 
Lukas Schynol
lukasschy96@gmail.com
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class SummaryWriterWrapper:
    def __init__(self, summary_dir):
        self.summary_dir = summary_dir
        self.data_path = os.path.join(summary_dir, "resultdata")
        self.tensorboard_writer = SummaryWriter(summary_dir)

        if os.path.isfile(self.data_path):
            self.data = torch.load(self.data_path)
        else:
            self.data = dict()

    def add_scalar(self, tag, scalar_value, globalstep=-1, walltime=None):
        self.tensorboard_writer.add_scalar(tag, scalar_value, globalstep, walltime=walltime)
        self.save_scalar(tag, scalar_value, globalstep, walltime=walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, globalstep=-1, walltime=None):
        self.tensorboard_writer.add_scalars(main_tag, tag_scalar_dict, globalstep, walltime=walltime)
        self.save_scalars(main_tag, tag_scalar_dict, globalstep, walltime=None)

    def add_figure(self, tag, figure, figure_data, global_step=-1, close=True, walltime=None):
        self.tensorboard_writer.add_figure(tag, figure, global_step=global_step, close=close, walltime=walltime)
        self.save_figure_data(tag, figure_data, global_step)

    def save_scalar(self, main_tag, scalar_value, globalstep, walltime=None):
        scalar_value = self._to_numpy(scalar_value)
        if not(walltime):
            walltime = time.time()
        ordered_dict = {"walltime": walltime, "step": globalstep, main_tag: scalar_value}
        if main_tag in self.data:
            # print(main_tag, scalar_value)
            row = pd.DataFrame(data=ordered_dict, index=[0])
            # self.data[main_tag] = self.data[main_tag].append(row)
            self.data[main_tag] = pd.concat([self.data[main_tag], row])
        else:
            # print(main_tag, scalar_value)
            self.data[main_tag] = pd.DataFrame(data=ordered_dict, index=[0])

    def save_scalars(self, main_tag, tag_scalar_dict, globalstep, walltime=None):
        tag_scalar_dict = self._to_numpy(tag_scalar_dict)
        if not(walltime):
            walltime = time.time()
        ordered_dict = {"walltime": walltime, "step": globalstep}
        for k, v in tag_scalar_dict.items():
            ordered_dict[k] = v
        if main_tag in self.data:
            row = pd.DataFrame(data=ordered_dict, index=[0])
            # self.data[main_tag] = self.data[main_tag].append(row)
            self.data[main_tag] = pd.concat([self.data[main_tag], row])
        else:
            self.data[main_tag] = pd.DataFrame(data=ordered_dict, index=[0])

    def save_figure_data(self, tag, figure_data, global_step):
        """
        :param tag:
        :param figure_data: dictionary containing iterables corresponding to axis data
        :param global_step:
        :return:
        """
        figure_data = self._to_numpy(figure_data)
        if tag in self.data:
            self.data[tag][global_step] = pd.DataFrame(data=figure_data)
        else:
            self.data[tag] = {global_step: pd.DataFrame(data=figure_data)}

    def _to_numpy(self, array_or_dict):
        if isinstance(array_or_dict, dict):
            for k, v in array_or_dict.items():
                v = self._to_numpy(v)
                array_or_dict[k] = v
            return array_or_dict
        else:
            if torch.is_tensor(array_or_dict):
                return array_or_dict.to(torch.device("cpu")).numpy()
            else:
                return np.array(array_or_dict)

    def flush(self):
        self.tensorboard_writer.flush()
        torch.save(self.data, self.data_path)

    def get_scalar_data(self, tag):
        if tag in self.data:
            tag_data = self.data[tag]
            if isinstance(tag_data, dict):  # for figures
                raise ValueError("Tag not found in data.")
            else:
                converted_data = {}
                for coln in tag_data.keys():
                    converted_data[coln] = tag_data[coln].to_numpy()
                return converted_data
        else:
            raise ValueError("Tag not found in data.")

    def export_data(self, tag=None):
        export_dir = os.path.join(self.summary_dir, "export")
        if not(os.path.isdir(export_dir)):
            os.mkdir(path=export_dir)

        if tag:
            if tag in self.data:
                tag_data = self.data[tag]
                if isinstance(tag_data, dict):  # for figures
                    figure_dir = os.path.join(export_dir, tag)
                    if not(os.path.isdir(figure_dir)):
                        os.mkdir(figure_dir)
                    for step, figure_data in tag_data.items():
                        name = tag + "_step_" + str(step)
                        self._write_to_csv(figure_data, name, figure_dir)
                else:
                    name = tag
                    self._write_to_csv(tag_data, name, export_dir)
            else:
                raise ValueError("Tag not found in data.")
        else:
            for tag in self.data.keys():
                # print(tag)
                # print(self.data[tag])
                self.export_data(tag=tag)

    def _write_to_csv(self, data, name, dir):
        path = os.path.join(dir, name + ".csv")
        data.to_csv(path_or_buf=path)


class DataExporter:
    def __init__(self, export_dir):
        self.export_dir = export_dir
        self.data = dict()

    def save_scalar(self, main_tag, scalar_value, globalstep, walltime=None):
        scalar_value = self._to_numpy(scalar_value)
        if not(walltime):
            walltime = time.time()
        if main_tag in self.data:
            row = pd.DataFrame(np.array([[walltime, globalstep, scalar_value]]),
                               columns=["walltime", "step", main_tag])
            # self.data[main_tag] = self.data[main_tag].append(row)
            self.data[main_tag] = pd.concat([self.data[main_tag], row])
        else:
            self.data[main_tag] = pd.DataFrame(np.array([[walltime, globalstep, scalar_value]]),
                                               columns=["walltime", "step", main_tag])

    def save_scalars(self, main_tag, tag_scalar_dict, globalstep, walltime=None):
        tag_scalar_dict = self._to_numpy(tag_scalar_dict)
        if not(walltime):
            walltime = time.time()
        ordered_dict = {"walltime": walltime, "step": globalstep}
        for k, v in tag_scalar_dict.items():
            ordered_dict[k] = v
        if main_tag in self.data:
            row = pd.DataFrame(data=ordered_dict, index=[0])
            # self.data[main_tag] = self.data[main_tag].append(row)
            self.data[main_tag] = pd.concat([self.data[main_tag], row])
        else:
            self.data[main_tag] = pd.DataFrame(data=ordered_dict, index=[0])

    def save_figure_data(self, tag, figure_data, global_step):
        """
        :param tag:
        :param figure_data: dictionary containing iterables corresponding to axis data
        :param global_step:
        :return:
        """
        figure_data = self._to_numpy(figure_data)
        if tag in self.data:
            self.data[tag][global_step] = pd.DataFrame(data=figure_data)
        else:
            self.data[tag] = {global_step: pd.DataFrame(data=figure_data)}

    def _to_numpy(self, array_or_dict):
        if isinstance(array_or_dict, dict):
            for k, v in array_or_dict.items():
                v = self._to_numpy(v)
                array_or_dict[k] = v
            return array_or_dict
        else:
            if torch.is_tensor(array_or_dict):
                return array_or_dict.to(torch.device("cpu")).numpy()
            else:
                return np.array(array_or_dict)

    def dump(self, tag=None):
        export_dir = self.export_dir
        if not(os.path.isdir(export_dir)):
            os.mkdir(path=export_dir)

        if tag:
            if tag in self.data:
                tag_data = self.data[tag]
                if isinstance(tag_data, dict):  # for figures
                    figure_dir = os.path.join(export_dir, tag)
                    if not(os.path.isdir(figure_dir)):
                        os.mkdir(figure_dir)
                    for step, figure_data in tag_data.items():
                        name = tag + "_step_" + str(step)
                        self._write_to_csv(figure_data, name, figure_dir)
                else:
                    name = tag
                    self._write_to_csv(tag_data, name, export_dir)
            else:
                raise ValueError("Tag not found in data.")
        else:
            for tag in self.data.keys():
                self.dump(tag=tag)

    def _write_to_csv(self, data, name, dir):
        path = os.path.join(dir, name + ".csv")
        data.to_csv(path_or_buf=path)

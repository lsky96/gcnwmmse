import os
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import comm.channel as channel
import comm.algorithm as algo
import comm.lossfun as lossfun
import comm.architectures as models
import numpy as np
import matplotlib.pyplot as plt
import datawriter
import time


class Trainer:
    def __init__(self,
                 model_name,
                 run,
                 timeout=23.9*3600,
                 rootdir=None,
                 measure_time=False,
                 ):
        # Prevent failures when saving if process is killed by SLURM
        self.timeout = timeout
        self.start_time = time.time()
        self.measure_time = measure_time
        self.time_meas_dict = {}

        self.run = run
        if rootdir is None:
            self.rootdir = os.getcwd()
        else:
            self.rootdir = rootdir
        self.dir = os.path.join(self.rootdir, model_name, "run_" + run)
        print(self.dir)
        self.trainer_config_path = os.path.join(self.dir, "trainer")
        self.model_config_path = os.path.join(self.dir, "model")
        self.state_path = os.path.join(self.dir, "state")
        self.testdata_path = os.path.join(self.dir, "testdata")  # default

        # Param
        self.i_step = 0

        # Init
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)
        self.writer = datawriter.SummaryWriterWrapper(self.dir)

    def __del__(self):
        if self.measure_time:
            self.dump_dict_as_txt("time_meas", self.time_meas_dict)

    def checkpoint_exists(self):
        return os.path.isfile(self.trainer_config_path) and os.path.isfile(self.model_config_path) and os.path.isfile(self.state_path)

    def load_last_checkpoint(self):
        raise NotImplementedError

    def save(self):
        pass

    def save_state(self, trainer_param, model_hyperparam, model_state_dict, optim_state_dict, scheduler_state_dict):
        print("\rSaving state...", end="", flush=True)
        trainer_param["i_step"] = self.i_step
        # trainer_param["i_sample"] = self.i_batch
        torch.save(trainer_param, self.trainer_config_path)
        torch.save(model_hyperparam, self.model_config_path)
        state_dicts = {
            "model": model_state_dict,
            "optim": optim_state_dict,
            "scheduler": scheduler_state_dict
        }
        torch.save(state_dicts, self.state_path)
        """
        if testdata_dict is None:
            pass
        else:
            torch.save(testdata_dict, self.testdata_path)
        """
        print("done!")

    def load_state(self, trainer_config_path=None, model_config_path=None, state_path=None, testdata_path=None):
        if trainer_config_path is None:
            trainer_config_path = self.trainer_config_path
        if model_config_path is None:
            model_config_path = self.model_config_path
        if state_path is None:
            state_path = self.state_path
        if testdata_path is None:
            testdata_path = self.testdata_path

        trainer_param = torch.load(trainer_config_path)
        self.i_step = trainer_param["i_step"]
        model_hyperparam = torch.load(model_config_path)
        state_dicts = torch.load(state_path)
        model_state_dict = state_dicts["model"]
        optim_state_dict = state_dicts["optim"]
        if "scheduler" in state_dicts:
            scheduler_state_dict = state_dicts["scheduler"]
        else:
            scheduler_state_dict = None

        if os.path.isfile(testdata_path):
            testdata_dict = torch.load(testdata_path)
        else:
            testdata_dict = None

        return trainer_param, model_hyperparam, model_state_dict, optim_state_dict, scheduler_state_dict, testdata_dict

    def load_testdata(self, testdata_path):
        testdata_path = os.path.join(self.rootdir, testdata_path)
        testdata_dict = torch.load(testdata_path)
        return testdata_dict

    def save_testdata(self, testdata_dict, testdata_path=None):
        if os.path.isfile(testdata_path):
            raise RuntimeError("File at testdata path already exists. Cannot overwrite testdata.")
        else:
            testdata_dir, _ = os.path.split(testdata_path)
            if not os.path.isdir(testdata_dir):
                os.mkdir(testdata_dir)
            torch.save(testdata_dict, self.testdata_path)

    def run_learning(self, num_iter):
        if self.measure_time:
            learningrun_starttime = time.time()

        for i_iter in range(num_iter):
            current_time = time.time()
            if current_time - self.start_time > self.timeout:
                break

            self.learning_iter()

        if self.measure_time:
            learningrun_time = time.time() - learningrun_starttime
            entry = "Learning run " + str(num_iter) + "[s]"
            self.time_meas_dict[entry] = learningrun_time

    def run_learning_upto_step(self, max_num_steps):
        num_iter_left = max_num_steps + 1 - self.i_step
        if num_iter_left > 0:
            self.run_learning(num_iter_left)
            # auto export data if done
            self.writer.export_data()
            self.write_finish_flag()
        else:
            print("Model already learned for {} steps.".format(max_num_steps))

    def learning_iter(self):
        # Call save here
        raise NotImplementedError

    def dump_dict_as_txt(self, name, meas_dict):
        outputfile_path = os.path.join(self.dir, name + ".txt")
        if os.path.isfile(outputfile_path):
            mode = "a"
        else:
            mode = "w"
        with open(outputfile_path, mode) as outputfile:
            for k, v in meas_dict.items():
                outputfile.write(str(k) + ": " + str(v) + "\n")

    def dump_info_text(self, infotext):
        outputfile_path = os.path.join(self.dir, "info" + ".txt")
        if os.path.isfile(outputfile_path):
            mode = "a"
        else:
            mode = "w"
        with open(outputfile_path, mode) as outputfile:
            outputfile.write(infotext + "\n")

    def write_finish_flag(self):
        superdir, _ = os.path.split(self.dir)
        finish_file_path = os.path.join(superdir, self.run + "_FINISHED.txt")
        with open(finish_file_path, "w") as outputfile:
            outputfile.write("Training finished after {} steps".format(self.i_step) + "\n")


class GeneralTrainer(Trainer):
    def __init__(self,
                 model_name,  # name of model
                 run,  # name of run
                 model_hyperparam=None,
                 learning_datagen_type="mimoifc_randcn",
                 learning_datagen_param=None,
                 learning_optimizer_param=None,
                 learning_scheduler_param=None,
                 learning_batch_size=None,
                 layer_parameter_schedule=None,
                 lossfun="rate",
                 lossfun_layers="all_excludeinit",
                 validation_every_num_step=10,
                 testdata_generate=False,
                 testdata_path=None,
                 testdata_gen_type="mimoifc_randcn",
                 testdata_gen_param=None,
                 testdata_batch_size=None,
                 testdata_num_iter=100,
                 testdata_num_inits=50,
                 testdata_mrc=True,
                 testdata_zf=False,
                 clip_grad_norm_val=None,
                 measure_time=True,
                 rootdir=None,
                 device=torch.device("cpu"),
                 ):
        """
        Creates a class that abstracts the model training.
        :param model_name: GCNWMMSE, GCNWMMSE_SISOAdhoc, UnfoldedPGD, IAIDNN, UWMMSE
        :param run: name of run
        :param model_hyperparam: dictionary of model parameters
        :param learning_datagen_type: mimoifc_randcn, mimoifc_triangle, siso_adhoc_2d, load_from_file (dict must contain
            keys path and num_samples)
        :param learning_datagen_param: dictionary containing params passed to training scenario factory function
        :param learning_optimizer_param: dictionary passed to AdamW optimizer
        :param learning_scheduler_param: dictionary passed to MultiStepLR scheduler
        :param learning_batch_size: iterable of int
        :param lossfun: rate, rate_samplenorm, iaidnn
        :param lossfun_layers: dict containing entries of form {stepcount:(list of layers)}, training layers for current
            training step are chosen from highest achieved stepcount
        :param validation_every_num_step:
        :param testdata_generate: set to True to generate testdata at step 0 at location testdata_path
        :param testdata_path: path to testdata
        :param testdata_gen_type: mimoifc_randcn, mimoifc_triangle, siso_adhoc_2d, deepmimo, load_from_file (dict must contain
            path and num_samples)
        :param testdata_gen_param: dictionary containing params passed to validation scenario factory function
        :param testdata_batch_size: iterable of int
        :param testdata_num_iter: number of WMMSE iterations to generate baseline results
        :param testdata_num_inits: number of WMMSE initialization to generate baseline results (WMMSE50)
        :param testdata_mrc: if True, WMMSE with MRC initialization baseline is generated
        :param testdata_zf: if True, WMMSE with ZF initialization baseline is generated
        :param clip_grad_norm_val: gradient norm clipping value
        :param measure_time: if True, generates a text data with elapsed time
        :param rootdir: root directory appended to all generated files (also testdata_path)
        :param device: torch.device to run calculations on
        """



        """
        Scheduler and optimizer params cannot be updated using the arguments.
        Arguments are overwritten from checkpoint data.
        """
        if learning_batch_size is None:
            learning_batch_size = [32]
        if testdata_batch_size is None:
            testdata_batch_size = [32]
        if model_hyperparam is None:
            model_hyperparam = {}
        if learning_optimizer_param is None:
            learning_optimizer_param = {"lr": 0.001}
        if learning_scheduler_param is None:
            learning_scheduler_param = {"milestones": []}

        super(GeneralTrainer, self).__init__(model_name, run, rootdir=rootdir, measure_time=measure_time)

        # Param
        self.device = device

        self.learned_num_samples = 0
        self.init_learning_rate = None
        self.clip_grad_norm_val = clip_grad_norm_val
        self.layer_parameter_schedule = layer_parameter_schedule
        self.lossfun = lossfun
        self.lossfun_layers = lossfun_layers
        self.validation_every_num_step = validation_every_num_step
        self.learning_datagen_type = learning_datagen_type
        self.learning_datagen_param = learning_datagen_param  # namedtuple with args, kwargs
        self.learning_batch_size = learning_batch_size

        # to be initialized and saved
        if testdata_path is not None and not testdata_generate:
            self.testdata_path = testdata_path
        else:
            testdata_generate = True
            if testdata_path is not None:
                self.testdata_path = testdata_path
            else:
                pass
                # self.testdatapath defaults to one of superclass

        self.testdata_dict = None
        self.testdata_mrc = testdata_mrc
        self.testdata_zf = testdata_zf

        self.model_name = model_name
        self.model_hyperparam = model_hyperparam
        self.learning_optimizer_param = learning_optimizer_param
        self.learning_scheduler_param = learning_scheduler_param

        # to be initialized
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.learningdata = None  # will only be initialized if learning_datagen_type=="load_from_file"
        self.learningdata_queue = []

        # Helper param
        self.consecutive_nan = 0

        if self.checkpoint_exists():
            print("\rLoading from checkpoint ...", end="", flush=True)
            model_state_dict, optim_state_dict, scheduler_state_dict = self.load_last_checkpoint()
            # Model init
            # print(self.model_hyperparam)
            self.model = models.get_model(self.model_name, **self.model_hyperparam, device=device)
            self.model.load_state_dict(model_state_dict)
            self.optimizer = optim.AdamW(self.model.parameters(), **self.learning_optimizer_param)
            self.optimizer.load_state_dict(optim_state_dict)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **self.learning_scheduler_param)
            self.scheduler.load_state_dict(scheduler_state_dict)
            print("Done!")
        else:
            """First init"""
            if not testdata_generate:  # argument was passed to constructor
                self.testdata_dict = self.load_testdata(self.testdata_path)
                print("Loaded test data.")
            else:
                if (self.testdata_path is not None) and os.path.isfile(self.testdata_path):
                    raise RuntimeError("File at testdata path already exists and may not be overwritten.")
                else:
                    self.testdata_dict = self.generate_test_data(testdata_gen_type, testdata_gen_param, testdata_batch_size, testdata_num_iter, testdata_num_inits, testdata_mrc, testdata_zf)
                    self.save_testdata(self.testdata_dict, testdata_path=self.testdata_path)
            self.model = models.get_model(model_name, **self.model_hyperparam, device=device)
            self.optimizer = optim.AdamW(self.model.parameters(), **self.learning_optimizer_param)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **self.learning_scheduler_param)
            self.save()

    def generate_test_data(self, data_gen_type, data_gen_param, batch_size, num_iter, num_inits, mrc, zf):
        if self.measure_time:
            gen_starttime = time.time()

        testdata_dict = {
            "data_gen_type": data_gen_type,
            "data_gen_param": data_gen_param,
            "batch_size": batch_size,
            "num_iter": num_iter,
            "num_inits": num_inits,
            "data": None,
            "results": None
        }

        print("\rData Initialization")
        testdata = self.generate_data(data_gen_type, data_gen_param, batch_size)
        """
        wrates_avg_layers, wrates_sopt, bss_pow_trial_avg_layers, bss_pow_ratio_trial_avg_layers, bss_pow_best_layers, bss_pow_ratio_best_layers = \
            algo.downlink_wmmse100(testdata, num_iter=num_iter, num_trials=100)
        """
        wrates_trial_avg_layers, wrates_best_layers, \
           bss_pow_trial_avg_layers, bss_pow_ratio_trial_avg_layers, \
           bss_pow_best_layers, bss_pow_ratio_best_layers = algo.large_scenario_set_downlink_wmmse50(testdata, num_iter=num_iter, num_trials=num_inits, use_stable=True)

        testdata_results = {
            "opt_layers": wrates_best_layers,
            "avg_layers": wrates_trial_avg_layers,
            "bss_pow_layers": bss_pow_trial_avg_layers,
            "bss_pow_ratio_layers": bss_pow_ratio_trial_avg_layers,
            "bss_pow_best_layers": bss_pow_best_layers,
            "bss_pow_ratio_best_layers": bss_pow_ratio_best_layers
        }

        # MRC Init
        if mrc:
            wrates_mrc_layers, bss_pow_mrc_layers, bss_pow_ratio_mrc_layers = \
                algo.large_scenario_set_initialized_downlink_wmmse(testdata, init="mrc", num_iter=num_iter, use_stable=True)

            testdata_results["mrc_layers"] = wrates_mrc_layers
            testdata_results["bss_pow_mrc_layers"] = bss_pow_mrc_layers
            testdata_results["bss_pow_ratio_mrc_layers"] = bss_pow_ratio_mrc_layers

        # Zero Forcing Init
        if zf:
            wrates_zf_layers, bss_pow_zf_layers, bss_pow_ratio_zf_layers = \
                algo.large_scenario_set_initialized_downlink_wmmse(testdata, init="zf", num_iter=num_iter, use_stable=True)

            testdata_results["zf_layers"] = wrates_zf_layers
            testdata_results["bss_pow_zf_layers"] = bss_pow_zf_layers
            testdata_results["bss_pow_ratio_zf_layers"] = bss_pow_ratio_zf_layers

        testdata_dict["data"] = testdata
        testdata_dict["results"] = testdata_results
        print(" Done!")

        if self.measure_time:
            gen_time = time.time() - gen_starttime
            self.time_meas_dict["Data Generation [s]"] = gen_time

        return testdata_dict

    def save(self):
        trainer_param = {
            "model_name": self.model_name,
            "testdata_path": self.testdata_path,
            "learned_num_samples": self.learned_num_samples,
            "learning_datagen_type": self.learning_datagen_type,
            "layer_parameter_schedule": self.layer_parameter_schedule,
            "learning_batch_size": self.learning_batch_size,  # required for learned num samples
            "learning_datagen_param": self.learning_datagen_param,
            "learning_optmizer_param": self.learning_optimizer_param,
            "learning_scheduler_param": self.learning_scheduler_param,
            "learningdata_queue": self.learningdata_queue,
            "lossfun": self.lossfun,
            "lossfun_layers": self.lossfun_layers,
            "validation_every_num_step": self.validation_every_num_step,
            "clip_grad_norm_val": self.clip_grad_norm_val,
            "testdata_mrc": self.testdata_mrc,
            "testdata_zf": self.testdata_zf
        }
        super(GeneralTrainer, self).save_state(trainer_param, self.model_hyperparam, self.model.state_dict(),
                                                        self.optimizer.state_dict(), self.scheduler.state_dict())

    def load_last_checkpoint(self):
        """Loads last checkpoint of trainer given the path."""
        trainer_param, model_hyperparam, model_state_dict, optim_state_dict, scheduler_state_dict, testdata_dict = \
            self.load_state()
        self.model_hyperparam = model_hyperparam
        self.testdata_dict = testdata_dict
        for k, v in trainer_param.items():
            setattr(self, k, v)

        return model_state_dict, optim_state_dict, scheduler_state_dict

    def get_lossfun_layers(self, step):
        """Returns list of layer indices to be used in the loss function for current training step."""
        if isinstance(self.lossfun_layers, dict):
            highest_milestone = 0
            for milestone in self.lossfun_layers.keys():
                if step >= milestone >= highest_milestone:
                    highest_milestone = milestone
            return self.lossfun_layers[highest_milestone]
        else:
            return self.lossfun_layers

    def generate_data(self, data_gen_type, data_gen_param, batch_size):
        if data_gen_type == "load_from_file":
            num_samples = self.learning_datagen_param["num_samples"]
            if self.learningdata is None:
                # Loads dataset, reduces set to num_samples
                learningdata_path = os.path.join(self.rootdir, self.learning_datagen_param["path"])
                learningdata = torch.load(learningdata_path)
                learningdata["data"] = channel.scenario_select_index(learningdata["data"], [list(range(num_samples))])
                self.learningdata = learningdata

            # Sampling
            if len(self.learningdata_queue) == 0:
                self.learningdata_queue = list(range(num_samples))
                random.shuffle(self.learningdata_queue)
            batch_size = min(batch_size[0], len(self.learningdata_queue))  # batch_size normally given as list
            current_indices = self.learningdata_queue[:batch_size]
            del self.learningdata_queue[:batch_size]

            return channel.scenario_select_index(self.learningdata["data"], [current_indices])

        else:
            if data_gen_type == "mimoifc_randcn":
                return channel.mimoifc_randcn(**data_gen_param, batch_size=batch_size, device=self.device)
            elif data_gen_type == "siso_adhoc_2d":
                return channel.siso_adhoc_2dscene(**data_gen_param, batch_size=batch_size, device=self.device)
            elif data_gen_type == "mimoifc_triangle" or data_gen_type == "mimotri":
                return channel.mimoifc_triangle(**data_gen_param, batch_size=batch_size, device=self.device)
            elif data_gen_type == "deepmimo":
                return channel.deepmimo(**data_gen_param, device=self.device)
            else:
                raise ValueError

    def apply_model(self, scenario):
        with torch.no_grad():
            self.model.eval()
            dl_beamformers, u, w, used_pow = self.model(scenario)

        return dl_beamformers, u, w, used_pow

    def learning_iter(self):
        if not(self.i_step == 0):
            self.model.train()
            # print(self.learning_datagen_type, self.learning_datagen_param)
            print("\rStep {}: {}".format(self.i_step, "Generating data..."), end="", flush=True)
            learning_batch_scenario = self.generate_data(self.learning_datagen_type, self.learning_datagen_param, self.learning_batch_size)

            print("\rStep {}: {}".format(self.i_step, "Forward pass..."), end="", flush=True)

            def forward_backward():
                dl_beamformers, _, _, _ = self.model(channel.prepare_csi(learning_batch_scenario))

                if self.lossfun == "rate":
                    loss = lossfun.wrate_loss(learning_batch_scenario, dl_beamformers, layers=self.get_lossfun_layers(self.i_step))
                elif self.lossfun == "rate_samplenorm":
                    loss = lossfun.wrate_loss_samplenorm(learning_batch_scenario, dl_beamformers, layers=self.get_lossfun_layers(self.i_step))
                elif self.lossfun == "iaidnn":
                    loss = lossfun.iaidnn_loss(learning_batch_scenario, dl_beamformers, layers=self.get_lossfun_layers(self.i_step))
                else:
                    raise ValueError
                    # print(loss)

                print("\rStep {}: {}".format(self.i_step, "Backward pass..."), end="", flush=True)
                loss.backward()

                return loss

            checkfornan = False
            if checkfornan:
                with torch.autograd.detect_anomaly():
                    loss = forward_backward()
            else:
                loss = forward_backward()

            # reject nan iterations:
            grad_is_nan = False
            for p in self.model.parameters():
                if p.grad is not None and torch.any(torch.isnan(p.grad)):
                    grad_is_nan = True
                    print("Grad was nan in step {}".format(self.i_step))
                    self.dump_info_text("Grad was nan in step {}".format(self.i_step))
                    break
                    # raise RuntimeError("Gradient was nan.")

            if grad_is_nan:
                if self.consecutive_nan >= 5:
                    self.dump_info_text("Gradient was nan more than 5 times.")
                    raise RuntimeError("Gradient was nan more than 5 times.")
                else:
                    self.consecutive_nan += 1
                    # restarting iteration
                    self.optimizer.zero_grad()
                    self.learning_iter()
                    return
            else:
                self.consecutive_nan = 0

            if self.clip_grad_norm_val is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_val)

            grad_debug = False
            if grad_debug:
                print(self.i_step)
                print("GRADIENTS")
                for n, p in self.model.named_parameters():
                    print(n)
                    print(p.grad)
                print("OPTIM PARAM")
                print(self.optimizer.state_dict())
                exit()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            self.learned_num_samples += math.prod(self.learning_batch_size)
            print("Done!")

        if self.i_step % self.validation_every_num_step == 0:
            self.model.eval()
            print("\rTesting performance ...", end="", flush=True)
            testdata = self.testdata_dict["data"]
            testdata_results = self.testdata_dict["results"]

            if self.i_step != 0:
                self.writer.add_scalar("Training Loss", loss.detach().numpy(), self.i_step)

            testdata_csi = channel.prepare_csi(testdata)
            with torch.no_grad():
                dl_beamformers, _, _, used_pow = self.model(testdata_csi)
                # print(used_pow.mean(-1).mean(-1))

            if self.model_name == "IAIDNN":
                _, wrate = channel.downlink_sum_rate_iaidnn(testdata, dl_beamformers)
            else:
                _, wrate = channel.downlink_sum_rate(testdata, dl_beamformers)

            model_avg_result_wrate = torch.mean(wrate[-1])
            used_pow_ratio = torch.mean(used_pow[-1] / testdata["bss_pow"])
            rate_plots_lastlayer = {
                "Unfolded WMMSE": model_avg_result_wrate,
            }
            if "avg_layers" in testdata_results:
                rate_plots_lastlayer["WMMSE 100 Trials Avg."] = testdata_results["avg_layers"][-1].mean()
                rate_plots_lastlayer["WMMSE 100 Trials Avg. Truncated"] = testdata_results["avg_layers"][self.model_hyperparam["num_layers"]].mean()

            if "opt_layers" in testdata_results:
                rate_plots_lastlayer["WMMSE 100 Trials Best"] = testdata_results["opt_layers"][-1].mean()

            if "mrc_layers" in testdata_results and self.testdata_mrc:
                # print(self.testdata_results["mrc_layers"].size())
                rate_plots_lastlayer["WMMSE MRC Init Avg."] = testdata_results["mrc_layers"][-1].mean()
                rate_plots_lastlayer["WMMSE MRC Init Avg. Truncated"] = testdata_results["mrc_layers"][self.model_hyperparam["num_layers"]].mean()

            if "zf_layers" in testdata_results and self.testdata_zf:
                # print(self.testdata_results["mrc_layers"].size())
                rate_plots_lastlayer["WMMSE ZF Init Avg."] = testdata_results["zf_layers"][-1].mean()
                rate_plots_lastlayer["WMMSE ZF Init Avg. Truncated"] = testdata_results["zf_layers"][self.model_hyperparam["num_layers"]].mean()

            self.writer.add_scalars("Test Set Avg. Rates", rate_plots_lastlayer, self.i_step)

            model_avg_result_wrate_layersum = torch.mean(torch.sum(wrate[1:], dim=0))
            rate_plots_layersum = {
                "Unfolded WMMSE": model_avg_result_wrate_layersum,
            }
            if "avg_layers" in testdata_results:
                rate_plots_layersum["WMMSE 100 Trials Avg. Trun."] = torch.sum(testdata_results["avg_layers"][1:(self.model_hyperparam["num_layers"]+1)], dim=0).mean()

            if "mrc_layers" in testdata_results and self.testdata_mrc:
                rate_plots_layersum["WMMSE MRC Init Avg. Trun."] = torch.sum(testdata_results["mrc_layers"][1:(self.model_hyperparam["num_layers"]+1)], dim=0).mean()

            if "zf_layers" in testdata_results and self.testdata_zf:
                rate_plots_layersum["WMMSE ZF Init Avg. Trun."] = torch.sum(testdata_results["zf_layers"][1:(self.model_hyperparam["num_layers"]+1)], dim=0).mean()

            self.writer.add_scalars("Test Set Avg. Rates SumLayers", rate_plots_layersum, self.i_step)

            power_plots = {"Unfolded WMMSE": used_pow_ratio}
            if "bss_pow_ratio_layers" in testdata_results:
                power_plots["WMMSE 100 Trials Avg."] = torch.mean(testdata_results["bss_pow_ratio_layers"][-1])
                power_plots["WMMSE 100 Trials Avg. Truncated"] = \
                    torch.mean(testdata_results["bss_pow_ratio_layers"][self.model_hyperparam["num_layers"]])  # last layer, avg over bss

            if "bss_pow_ratio_mrc_layers" in testdata_results and self.testdata_mrc:
                # print(self.testdata_results["bss_pow_ratio_mrc_layers"].size())
                power_plots["WMMSE MRC Init Avg."] = torch.mean(testdata_results["bss_pow_ratio_mrc_layers"][-1])  # last layer, avg over bss
                power_plots["WMMSE MRC Init Avg. Truncated"] = \
                    torch.mean(testdata_results["bss_pow_ratio_mrc_layers"][self.model_hyperparam["num_layers"]])

            if "bss_pow_ratio_zf_layers" in testdata_results and self.testdata_zf:
                # print(self.testdata_results["bss_pow_ratio_mrc_layers"].size())
                power_plots["WMMSE ZF Init Avg."] = torch.mean(testdata_results["bss_pow_ratio_zf_layers"][-1])  # last layer, avg over bss
                power_plots["WMMSE ZF Init Avg. Truncated"] = \
                    torch.mean(testdata_results["bss_pow_ratio_zf_layers"][self.model_hyperparam["num_layers"]])

            if "bss_pow_ratio_best_layers" in testdata_results:
                # print(self.testdata_results["bss_pow_ratio_mrc_layers"].size())
                power_plots["WMMSE Best Init Avg."] = torch.mean(testdata_results["bss_pow_ratio_best_layers"][-1])  # last layer, avg over bss
                power_plots["WMMSE Best Init Avg. Truncated"] = \
                    torch.mean(testdata_results["bss_pow_ratio_best_layers"][self.model_hyperparam["num_layers"]])

            self.writer.add_scalars("Used Power Budget Ratio", power_plots, self.i_step)

            # avg rate over iteration
            if self.i_step % (self.validation_every_num_step*5) == 0:
                num_layers_winit = list(wrate.size())[0]
                layer_ax = torch.arange(0, num_layers_winit).cpu().numpy()

                model_wrate_over_layers = wrate.view(num_layers_winit, -1).mean(1)
                for_plotting = [layer_ax, model_wrate_over_layers.cpu().numpy(), "-*", layer_ax,
                        testdata_results["avg_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy(), "-+"]
                legend = ["Unfolded WMMSE", "WMMSE Rand. Init"]
                plotdata = {"Layer": layer_ax,
                            "Unfolded WMMSE": model_wrate_over_layers.cpu().numpy(),
                            "WMMSE Rand. Init": testdata_results["avg_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy()}

                if "mrc_layers" in testdata_results and self.testdata_mrc:
                    for_plotting.extend([layer_ax, testdata_results["mrc_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy(), "-x"])
                    legend.append("WMMSE MRC Init")
                    plotdata["WMMSE MRC Init"] = testdata_results["mrc_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy()

                if "zf_layers" in testdata_results and self.testdata_zf:
                    for_plotting.extend([layer_ax, testdata_results["zf_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy(), "-|"])
                    legend.append("WMMSE ZF Init")
                    plotdata["WMMSE ZF Init"] = testdata_results["zf_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy()

                if "opt_layers" in testdata_results:
                    for_plotting.extend([layer_ax, testdata_results["opt_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy(), "-_"])
                    legend.append("WMMSE Best Init")
                    plotdata["WMMSE Best Init"] = testdata_results["opt_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(1).cpu().numpy()
                fig, ax = plt.subplots(1, 1)
                ax.plot(*for_plotting)
                ax.grid()
                ax.legend(legend)
                ax.set_ylim(bottom=0)
                ax.set(xlabel='Layer', ylabel='Avg. Rate')
                self.writer.add_figure("Avg. Rate over Layers", fig, plotdata, global_step=self.i_step)

            print("Done!")

        self.i_step += 1
        # Saving
        if self.i_step % 10 == 1:
            self.save()
            self.writer.flush()

    def evaluate_on_test_set(self, path):
        """
        Evaluates model on test set in path and saves results to <rundir>\<testdataname>.
        :param path: testdata directory
        """
        dir, testdata_name = os.path.split(path)
        # _, dir_name = os.path.split(dir)

        export_dir = os.path.join(self.dir, testdata_name)
        exporter = datawriter.DataExporter(export_dir)

        data_dict = torch.load(path)
        system_data = data_dict["data"]
        csi = channel.prepare_csi(system_data)
        system_results = data_dict["results"]

        with torch.no_grad():
            dl_beamformers, _, _, used_pow = self.model(csi)

        if self.model_name == "IAIDNN":
            _, wrate = channel.downlink_sum_rate_iaidnn(system_data, dl_beamformers)
        else:
            _, wrate = channel.downlink_sum_rate(system_data, dl_beamformers)
        model_avg_result_wrate = torch.mean(wrate[-1])
        used_pow_ratio = torch.mean(used_pow[-1] / system_data["bss_pow"])
        rate_plots_lastlayer = {
            "Unfolded WMMSE": model_avg_result_wrate,
        }
        if "avg_layers" in system_results:
            rate_plots_lastlayer["WMMSE 100 Trials Avg."] = system_results["avg_layers"][-1].mean()
            rate_plots_lastlayer["WMMSE 100 Trials Avg. Truncated"] = system_results["avg_layers"][
                self.model_hyperparam["num_layers"]].mean()

        if "opt_layers" in system_results:
            rate_plots_lastlayer["WMMSE 100 Trials Best"] = system_results["opt_layers"][-1].mean()

        if "mrc_layers" in system_results and self.testdata_mrc:
            rate_plots_lastlayer["WMMSE MRC Init Avg."] = system_results["mrc_layers"][-1].mean()
            rate_plots_lastlayer["WMMSE MRC Init Avg. Truncated"] = system_results["mrc_layers"][
                self.model_hyperparam["num_layers"]].mean()

        if "zf_layers" in system_results and self.testdata_zf:
            rate_plots_lastlayer["WMMSE ZF Init Avg."] = system_results["zf_layers"][-1].mean()
            rate_plots_lastlayer["WMMSE ZF Init Avg. Truncated"] = system_results["zf_layers"][
                self.model_hyperparam["num_layers"]].mean()

        exporter.save_scalars("Test Set Avg. Rates", rate_plots_lastlayer, self.i_step)

        model_avg_result_wrate_layersum = torch.mean(torch.sum(wrate[1:], dim=0))
        rate_plots_layersum = {
            "Unfolded WMMSE": model_avg_result_wrate_layersum,
        }
        if "avg_layers" in system_results:
            rate_plots_layersum["WMMSE 100 Trials Avg. Trun."] = torch.sum(
                system_results["avg_layers"][1:(self.model_hyperparam["num_layers"] + 1)], dim=0).mean()

        if "mrc_layers" in system_results and self.testdata_mrc:
            rate_plots_layersum["WMMSE MRC Init Avg. Trun."] = torch.sum(
                system_results["mrc_layers"][1:(self.model_hyperparam["num_layers"] + 1)], dim=0).mean()

        if "zf_layers" in system_results and self.testdata_zf:
            rate_plots_layersum["WMMSE ZF Init Avg. Trun."] = torch.sum(
                system_results["zf_layers"][1:(self.model_hyperparam["num_layers"] + 1)], dim=0).mean()

        exporter.save_scalars("Test Set Avg. Rates SumLayers", rate_plots_layersum, self.i_step)

        power_plots = {"Unfolded WMMSE": used_pow_ratio}
        if "bss_pow_ratio_layers" in system_results:
            power_plots["WMMSE 100 Trials Avg."] = torch.mean(system_results["bss_pow_ratio_layers"][-1])
            power_plots["WMMSE 100 Trials Avg. Truncated"] = \
                torch.mean(system_results["bss_pow_ratio_layers"][
                               self.model_hyperparam["num_layers"]])  # last layer, avg over bss

        if "bss_pow_ratio_mrc_layers" in system_results and self.testdata_mrc:
            # print(self.testdata_results["bss_pow_ratio_mrc_layers"].size())
            power_plots["WMMSE MRC Init Avg."] = torch.mean(
                system_results["bss_pow_ratio_mrc_layers"][-1])  # last layer, avg over bss
            power_plots["WMMSE MRC Init Avg. Truncated"] = \
                torch.mean(system_results["bss_pow_ratio_mrc_layers"][self.model_hyperparam["num_layers"]])

        if "bss_pow_ratio_zf_layers" in system_results and self.testdata_zf:
            # print(self.testdata_results["bss_pow_ratio_mrc_layers"].size())
            power_plots["WMMSE ZF Init Avg."] = torch.mean(
                system_results["bss_pow_ratio_zf_layers"][-1])  # last layer, avg over bss
            power_plots["WMMSE ZF Init Avg. Truncated"] = \
                torch.mean(system_results["bss_pow_ratio_zf_layers"][self.model_hyperparam["num_layers"]])

        if "bss_pow_ratio_best_layers" in system_results:
            # print(self.testdata_results["bss_pow_ratio_mrc_layers"].size())
            power_plots["WMMSE Best Init Avg."] = torch.mean(
                system_results["bss_pow_ratio_best_layers"][-1])  # last layer, avg over bss
            power_plots["WMMSE Best Init Avg. Truncated"] = \
                torch.mean(system_results["bss_pow_ratio_best_layers"][self.model_hyperparam["num_layers"]])

        exporter.save_scalars("Used Power Budget Ratio", power_plots, self.i_step)

        # avg rate over iteration

        num_layers_winit = list(wrate.size())[0]
        layer_ax = torch.arange(0, num_layers_winit).cpu().numpy()

        model_wrate_over_layers = wrate.view(num_layers_winit, -1).mean(1)
        for_plotting = [layer_ax, model_wrate_over_layers.cpu().numpy(), "-*", layer_ax,
                        system_results["avg_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(
                            1).cpu().numpy(), "-+"]
        legend = ["Unfolded WMMSE", "WMMSE Rand. Init"]
        plotdata = {"Layer": layer_ax,
                    "Unfolded WMMSE": model_wrate_over_layers.cpu().numpy(),
                    "WMMSE Rand. Init": system_results["avg_layers"][0:num_layers_winit].view(num_layers_winit,
                                                                                              -1).mean(1).cpu().numpy()}

        if "mrc_layers" in system_results and self.testdata_mrc:
            for_plotting.extend([layer_ax,
                                 system_results["mrc_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(
                                     1).cpu().numpy(), "-x"])
            legend.append("WMMSE MRC Init")
            plotdata["WMMSE MRC Init"] = system_results["mrc_layers"][0:num_layers_winit].view(num_layers_winit,
                                                                                               -1).mean(1).cpu().numpy()

        if "zf_layers" in system_results and self.testdata_zf:
            for_plotting.extend([layer_ax,
                                 system_results["zf_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(
                                     1).cpu().numpy(), "-|"])
            legend.append("WMMSE ZF Init")
            plotdata["WMMSE ZF Init"] = system_results["zf_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(
                1).cpu().numpy()

        if "opt_layers" in system_results:
            for_plotting.extend([layer_ax,
                                 system_results["opt_layers"][0:num_layers_winit].view(num_layers_winit, -1).mean(
                                     1).cpu().numpy(), "-_"])
            legend.append("WMMSE Best Init")
            plotdata["WMMSE Best Init"] = system_results["opt_layers"][0:num_layers_winit].view(num_layers_winit,
                                                                                                -1).mean(
                1).cpu().numpy()

        exporter.save_figure_data("Avg. Rate over Layers", plotdata, global_step=self.i_step)

        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)
        exporter.dump()
        # self.get_rate_distribution(path, num_bins=20, user=5)  # also dumps data

        print("Tested on", testdata_name)

    def get_rate_distribution(self, path, num_bins=10, binrange="auto", user="sum"):
        # Simplifies making histograms
        dir, _ = os.path.split(path)
        _, dir_name = os.path.split(dir)

        export_dir = os.path.join(self.dir, dir_name + "_hist")

        data_dict = torch.load(path)
        system_data = data_dict["data"]
        system_data = channel.scenario_select_index(system_data, [0])

        with torch.no_grad():
            dl_beamformers_model, _, _, used_pow = self.model(system_data)

        if self.model_name == "IAIDNN":
            _, wrate_model = channel.downlink_rate_iaidnn(system_data, dl_beamformers_model)
        else:
            _, wrate_model = channel.downlink_rate(system_data, dl_beamformers_model)

        if "results" not in data_dict or True:
            dl_beamformers_wmmse_mrc, _, _, _ = algo.downlink_wmmse_stable(system_data,
                                                                           init_dl_beamformer=algo.downlink_mrc(
                                                                               system_data),
                                                                           num_iter=100)

            _, wrate_wmmse_mrc_raw = channel.downlink_rate(system_data, dl_beamformers_wmmse_mrc)
            include_all = False
        else:
            results = data_dict["results"]
            wrate_wmmse_best_raw = results["opt_layers"]
            wrate_wmmse_avg_raw = results["avg_layers"]
            wrate_wmmse_mrc_raw = results["mrc_layers"]
            include_all = True

        if user == "sum":
            wrate_model = torch.sum(wrate_model[-1], dim=-1)
            if not include_all:
                wrate_wmmse_mrc = torch.sum(wrate_wmmse_mrc_raw[-1], dim=-1)
                wrate_wmmse_mrc_trun = torch.sum(wrate_wmmse_mrc_raw[self.model.num_layers], dim=-1)
            else:
                wrate_wmmse_mrc = wrate_wmmse_mrc_raw[-1]
                wrate_wmmse_mrc_trun = wrate_wmmse_mrc_raw[self.model.num_layers]
        else:
            wrate_model = wrate_model[-1, ..., user]
            wrate_wmmse_mrc = wrate_wmmse_mrc_raw[-1, ..., user]
            wrate_wmmse_mrc_trun = wrate_wmmse_mrc_raw[self.model.num_layers, ..., user]

        if include_all:
            if user == "sum":
                wrate_wmmse_best = wrate_wmmse_best_raw[-1]
                wrate_wmmse_avg = wrate_wmmse_avg_raw[-1]
                wrate_wmmse_avg_trun = wrate_wmmse_avg_raw[self.model.num_layers]
            else:
                raise ValueError

        wrate_model = wrate_model.cpu().numpy()
        wrate_wmmse_mrc = wrate_wmmse_mrc.cpu().numpy()
        wrate_wmmse_mrc_trun = wrate_wmmse_mrc_trun.cpu().numpy()

        if not include_all:
            rangemin = np.min(np.concatenate((wrate_model, wrate_wmmse_mrc, wrate_wmmse_mrc_trun)))
            rangemax = np.max(np.concatenate((wrate_model, wrate_wmmse_mrc, wrate_wmmse_mrc_trun)))
        else:
            wrate_wmmse_best = wrate_wmmse_best.cpu().numpy()
            wrate_wmmse_avg = wrate_wmmse_avg.cpu().numpy()
            wrate_wmmse_avg_trun = wrate_wmmse_avg_trun.cpu().numpy()

            # print(wrate_model.shape, wrate_wmmse_avg.shape, wrate_wmmse_avg_trun.shape)
            rangemin = np.min(np.concatenate((wrate_model, wrate_wmmse_avg, wrate_wmmse_avg_trun, wrate_wmmse_best,
                                              wrate_wmmse_mrc, wrate_wmmse_mrc_trun)))
            rangemax = np.max(np.concatenate((wrate_model, wrate_wmmse_avg, wrate_wmmse_avg_trun, wrate_wmmse_best,
                                              wrate_wmmse_mrc, wrate_wmmse_mrc_trun)))

        if binrange == "auto":
            binrange = (rangemin - 1, rangemax + 1)

        """
        for density in (False, True):
            hist_model, bin_edges_model = np.histogram(wrate_model, bins=num_bins, range=range, density=density)
            hist_wmmse, bin_edges_wmmse= np.histogram(wrate_wmmse, bins=num_bins, range=range, density=density)
            hist_wmmse_trun, bin_edges_wmmse_trun = np.histogram(wrate_wmmse_trun, bins=num_bins, range=range, density=density)
        """

        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)

        hist_model, bin_edges_model = np.histogram(wrate_model, bins=num_bins, range=binrange, density=False)
        path_hist_model = os.path.join(export_dir, "model" + "_hist" + "_user" + str(user) + ".csv")
        np.savetxt(path_hist_model, np.stack([np.concatenate([hist_model, np.zeros(1)]), bin_edges_model]),
                   delimiter=",", fmt="%.3f")

        path_model = os.path.join(export_dir, "model" + "_user" + str(user) + ".csv")
        np.savetxt(path_model, wrate_model, delimiter=",")

        path_wmmse_mrc = os.path.join(export_dir, "wmmse_mrc" + "_user" + str(user) + ".csv")
        np.savetxt(path_wmmse_mrc, wrate_wmmse_mrc, delimiter=",")

        path_wmmse_mrc_trun = os.path.join(export_dir, "wmmse_mrc_trun" + "_user" + str(user) + ".csv")
        np.savetxt(path_wmmse_mrc_trun, wrate_wmmse_mrc_trun, delimiter=",")

        if include_all:
            hist_wmmse_best, bin_edges_wmmse_best = np.histogram(wrate_wmmse_best, bins=num_bins, range=binrange,
                                                                 density=False)
            path_hist_wmmse_best = os.path.join(export_dir, "wmmse_best" + "_hist" + "_user" + str(user) + ".csv")
            np.savetxt(path_hist_wmmse_best,
                       np.stack([np.concatenate([hist_wmmse_best, np.zeros(1)]), bin_edges_wmmse_best]), delimiter=",",
                       fmt="%.3f")

            hist_wmmse_avg, bin_edges_wmmse_avg = np.histogram(wrate_wmmse_avg, bins=num_bins, range=binrange,
                                                               density=False)
            path_hist_wmmse_avg = os.path.join(export_dir, "wmmse_avg" + "_hist" + "_user" + str(user) + ".csv")
            np.savetxt(path_hist_wmmse_avg,
                       np.stack([np.concatenate([hist_wmmse_avg, np.zeros(1)]), bin_edges_wmmse_avg]), delimiter=",",
                       fmt="%.3f")

            hist_wmmse_mrc, bin_edges_wmmse_mrc = np.histogram(wrate_wmmse_mrc, bins=num_bins, range=binrange,
                                                               density=False)
            path_hist_wmmse_mrc = os.path.join(export_dir, "wmmse_mrc" + "_hist" + "_user" + str(user) + ".csv")
            np.savetxt(path_hist_wmmse_mrc,
                       np.stack([np.concatenate([hist_wmmse_mrc, np.zeros(1)]), bin_edges_wmmse_mrc]), delimiter=",",
                       fmt="%.3f")

            path_wmmse_avg = os.path.join(export_dir, "wmmse_avg" + "_user" + str(user) + ".csv")
            np.savetxt(path_wmmse_avg, wrate_wmmse_avg, delimiter=",")

            path_wmmse_avg_trun = os.path.join(export_dir, "wmmse_avg_trun" + "_user" + str(user) + ".csv")
            np.savetxt(path_wmmse_avg_trun, wrate_wmmse_avg_trun, delimiter=",")

            path_wmmse_best = os.path.join(export_dir, "wmmse_best" + "_user" + str(user) + ".csv")
            np.savetxt(path_wmmse_best, wrate_wmmse_best, delimiter=",")


class DataSetGenerator:
    """Class to generate test data"""
    def __init__(self, device=torch.device("cpu")):
        self.device = device

    def generate_training_data_file(self, dir, data_gen_type, data_gen_param, size):
        dir = os.path.abspath(dir)
        if not os.path.isdir(dir):
            os.mkdir(dir)

        path = os.path.join(dir, "trainingdata")
        if os.path.isfile(path):
            raise RuntimeError("File does already exist. Cannot overwrite.")

        data = self.generate_data(data_gen_type, data_gen_param, size)
        trainingdata_dict = {
            "data_gen_type": data_gen_type,
            "data_gen_param": data_gen_param,
            "batch_size": size,
            "data": data
        }

        torch.save(trainingdata_dict, path)

    def generate_test_data_file(self, dir, data_gen_type, data_gen_param, batch_size, num_iter, mrc, zf):
        """
        Creates test data dictionary and saves it at location dir
        :param dir: save location
        Other parameters described in generate_test_data
        """
        dir = os.path.abspath(dir)
        if not os.path.isdir(dir):
            os.mkdir(dir)

        path = os.path.join(dir, "trainingdata")
        if os.path.isfile(path):
            raise RuntimeError("File does already exist. Cannot overwrite.")

        testdata_dict = self.generate_test_data(data_gen_type, data_gen_param, batch_size, num_iter, mrc, zf)

        torch.save(testdata_dict, path)
        print("Data written.")

    def generate_test_data(self, data_gen_type, data_gen_param, batch_size, num_iter, mrc, zf):
        """
        Creates test data dictionary based on parameters.
        :param data_gen_type: mimoifc_randcn, mimoifc_triangle, siso_adhoc_2d, load_from_file (dict must contain
            path and num_samples)
        :param data_gen_param: dictionary containing params passed to validation scenario factory function
        :param data_batch_size: iterable of int
        :param data_num_iter: number of WMMSE iterations to generate baseline results
        :param mrc: if True, WMMSE with MRC initialization baseline is generated
        :param zf: if True, WMMSE with ZF initialization baseline is generated
        :return: testdata dictionary
        """
        testdata_dict = {
            "data_gen_type": data_gen_type,
            "data_gen_param": data_gen_param,
            "batch_size": batch_size,
            "num_iter": num_iter,
            "data": None,
            "results": None
        }

        print("\rData Initialization")
        testdata = self.generate_data(data_gen_type, data_gen_param, batch_size)
        wrates_trial_avg_layers, wrates_best_layers, \
           bss_pow_trial_avg_layers, bss_pow_ratio_trial_avg_layers, \
           bss_pow_best_layers, bss_pow_ratio_best_layers = algo.large_scenario_set_downlink_wmmse50(testdata,
                                                                                                     num_iter=num_iter,
                                                                                                     num_trials=50)

        testdata_results = {
            "opt_layers": wrates_best_layers,
            "avg_layers": wrates_trial_avg_layers,
            "bss_pow_layers": bss_pow_trial_avg_layers,
            "bss_pow_ratio_layers": bss_pow_ratio_trial_avg_layers,
            "bss_pow_best_layers": bss_pow_best_layers,
            "bss_pow_ratio_best_layers": bss_pow_ratio_best_layers
        }

        # MRC Init
        if mrc:
            wrates_mrc_layers, bss_pow_mrc_layers, bss_pow_ratio_mrc_layers = \
                algo.large_scenario_set_initialized_downlink_wmmse(testdata, init="mrc", num_iter=num_iter)

            testdata_results["mrc_layers"] = wrates_mrc_layers
            testdata_results["bss_pow_mrc_layers"] = bss_pow_mrc_layers
            testdata_results["bss_pow_ratio_mrc_layers"] = bss_pow_ratio_mrc_layers

        # Zero Forcing Init
        if zf:
            wrates_zf_layers, bss_pow_zf_layers, bss_pow_ratio_zf_layers = \
                algo.large_scenario_set_initialized_downlink_wmmse(testdata, init="zf", num_iter=num_iter)

            testdata_results["zf_layers"] = wrates_zf_layers
            testdata_results["bss_pow_zf_layers"] = bss_pow_zf_layers
            testdata_results["bss_pow_ratio_zf_layers"] = bss_pow_ratio_zf_layers

        testdata_dict["data"] = testdata
        testdata_dict["results"] = testdata_results
        print(" Done!")

        return testdata_dict

    def generate_data(self, data_gen_type, data_gen_param, batch_size):
        args, kwargs = data_gen_param.values()
        if data_gen_type == "mimoifc_randcn":
            data = channel.mimoifc_randcn(*args, **kwargs, batch_size=batch_size, device=self.device)
        elif data_gen_type == "siso_adhoc_2d":
            data = channel.siso_adhoc_2dscene(*args, **kwargs, batch_size=batch_size, device=self.device)
        elif data_gen_type == "mimoifc_triangle" or data_gen_type == "mimotri":
            data = channel.mimoifc_triangle(*args, **kwargs, batch_size=batch_size, device=self.device)
        elif data_gen_type == "deepmimo":
            return channel.deepmimo(*args, **kwargs, device=self.device)
        else:
            raise ValueError

        return data

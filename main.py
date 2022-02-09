import sys
import os
import comm.mathutil as util
import torch
import comm.trainer as trainer
import comm.channel
import comm.algorithm as algo

DATADIR = os.path.join(os.getcwd(), "data")


def gcnwmmse_example_training():
    """
    Note that depending on the values for testdata_num_iter, testdata_num_inits, generating data
    """
    max_num_training_steps = 100
    run_name = "gcnwmmse_example"
    model_name = "GCNWMMSE"
    testdata_path = os.path.join(DATADIR, "testdata", "exampledata_3BSs_9UEs")

    model_parameters = {
            "num_layers": 7,
            "diaglnorm": False,
            "biasnorm": True,
            "w_poly": {"degree": 4, "norm_above": 1},
            "lagrangian_max_iter": 8,
            "v_num_channels": 4,
            "v_active_components": {"ar": True, "ma": False, "diagload": True, "nonlin": True, "bias": True},
            "v_bypass_param": {"position": "before_nonlin"},
            "shared_parameters": False,
        }
    data_gen_type = "mimoifc_randcn"
    data_gen_param = {
        "bss_dim": [12, 12, 12],
        "users_dim": [*[2] * 9],
        "assignment": [*[0] * 3, *[1] * 3, *[2] * 3],
        "snr": 20,
        "channel_gain": 1,
    }

    optimizer_param = {
        "lr": 0.01,
        "weight_decay": 1e-3,
        "betas": (0.9, 0.99)
    }

    scheduler_param = {
        "milestones": [2500, 5000, 7500],
        "gamma": 1 / 10,
    }

    # initialize run
    t = trainer.GeneralTrainer(
                 model_name=model_name,
                 run=run_name,
                 model_hyperparam=model_parameters,
                 learning_datagen_type=data_gen_type,
                 learning_datagen_param=data_gen_param,
                 learning_optimizer_param=optimizer_param,
                 learning_scheduler_param=scheduler_param,
                 learning_batch_size=[100],
                 layer_parameter_schedule=None,
                 lossfun="rate_samplenorm",
                 lossfun_layers=[-1],
                 testdata_generate=True,
                 testdata_path=testdata_path,
                 testdata_gen_type=data_gen_type,
                 testdata_gen_param=data_gen_param,
                 testdata_batch_size=[10],
                 testdata_num_iter=100,
                 testdata_num_inits=50,
                 clip_grad_norm_val=1,
                 measure_time=True,
                 rootdir=DATADIR,
                 device=torch.device("cpu"),
                 )
    t.run_learning_upto_step(max_num_training_steps)
    # Upon completion, trainer class will export training statistics into run dir, otherwise use tensorboard.

    t.evaluate_on_test_set(testdata_path)


if __name__ == "__main__":
    gcnwmmse_example_training()

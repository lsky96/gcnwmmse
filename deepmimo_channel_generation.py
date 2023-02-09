import os
import numpy as np
import DeepMIMO
import pickle

"""
Deep MIMO raw files must be placed into folder deepmimo/raw_scenes.
"""


def generate_scenario61():
    # Return to old scenario BS
    active_bs = np.array([13, 14, 15, 16])
    active_rows = [2752, 3600]
    split1, split2 = [0, 1], [2, 3]

    trainingdata_setsize = 5000
    testdata_setsize = 1000
    in_name = 'O1_60'
    out_name = "scenario61"

    num_bss = 4
    num_users = 4
    num_tx_ant = 12
    num_rx_ant = 4

    input_path = os.path.join("deepmimo", "raw_scenes")
    output_dir = os.path.join("deepmimo", "channel_outputs")

    # Set the main folder containing extracted scenarios
    parameters = DeepMIMO.default_params()
    parameters['scenario'] = in_name
    parameters['dataset_folder'] = input_path

    parameters['active_BS'] = active_bs
    parameters['user_row_first'] = active_rows[0]
    parameters['user_row_last'] = active_rows[1]

    parameters['num_paths'] = 10

    parameters['bs_antenna']['shape'] = np.array([1, 3, 4])
    parameters['bs_antenna']['spacing'] = 0.5

    parameters['ue_antenna']['shape'] = np.array([1, 2, 2])
    parameters['ue_antenna']['spacing'] = 0.5

    parameters['enable_BS2BS'] = 0

    parameters['OFDM_channels'] = 1  # Frequency (OFDM) or time domain channels
    parameters['OFDM']['subcarriers'] = 1
    parameters['OFDM']['subcarriers_limit'] = 1
    parameters['OFDM']['subcarriers_sampling'] = 1
    parameters['OFDM']['bandwidth'] = 240e-6  # in GHz

    # Generate data
    dataset = DeepMIMO.generate_data(parameters)

    # Choose userindices
    num_user_pos = dataset[0]["user"]["channel"].shape[0]
    # print(dataset[0]["user"]["channel"].shape)
    bs_allowed_idx = []
    bs_num_allowed_idx = []
    num_users_per_bs = num_users // num_bss
    for k in range(num_bss):
        idx = ((np.abs(dataset[k]["user"]["channel"]).sum() > 1e-15)*np.arange(num_user_pos)).astype(int)
        bs_allowed_idx.append(idx)
        bs_num_allowed_idx.append(len(idx))

    training_user_pos_idx = np.zeros([trainingdata_setsize, num_users], dtype=int)
    for i in range(trainingdata_setsize):
        for k in range(num_bss):
            if k in split1:
                st, en = 0, bs_num_allowed_idx[k] // 2
            elif k in split2:
                st, en = bs_num_allowed_idx[k] // 2, None
            else:
                st, en = 0, None
            training_user_pos_idx[i][(k*num_users_per_bs):((k+1)*num_users_per_bs)] = \
                np.random.choice(bs_allowed_idx[k][st:en:2], num_users_per_bs, replace=False)

    test_user_pos_idx = np.zeros([testdata_setsize, num_users], dtype=int)
    for i in range(testdata_setsize):
        for k in range(num_bss):
            if k in split1:
                st, en = 0, bs_num_allowed_idx[k] // 2
            elif k in split2:
                st, en = bs_num_allowed_idx[k] // 2, None
            else:
                st, en = 0, None
            test_user_pos_idx[i][(k*num_users_per_bs):((k+1)*num_users_per_bs)] = \
                np.random.choice(bs_allowed_idx[k][(st+1):en:2], num_users_per_bs, replace=False)

    training_channels = []
    test_channels = []

    for k in range(num_bss):
        training_channels.append([])
        test_channels.append([])
        for i in range(num_users):
            H = np.zeros([trainingdata_setsize, num_rx_ant, num_tx_ant], dtype=complex)
            for s in range(trainingdata_setsize):
                H[s] = dataset[k]["user"]["channel"][training_user_pos_idx[s, i]].squeeze(axis=-1)
            training_channels[k].append(H)

            H = np.zeros([testdata_setsize, num_rx_ant, num_tx_ant], dtype=complex)
            for s in range(testdata_setsize):
                H[s] = dataset[k]["user"]["channel"][test_user_pos_idx[s, i]].squeeze(axis=-1)
            test_channels[k].append(H)

    with open(os.path.join(output_dir, out_name+"_training.pkl"), mode="wb") as fh:
        pickle.dump(training_channels, fh)

    with open(os.path.join(output_dir, out_name + "_test.pkl"), mode="wb") as fh:
        pickle.dump(test_channels, fh)


def generate_scenario63():
    # Return to old scenario BS
    active_bs = np.array([13, 14, 15, 16])
    active_rows = [3300, 3400]
    split1, split2 = [0, 1], [2, 3]

    trainingdata_setsize = 5000
    testdata_setsize = 1000
    in_name = 'O1_60'
    out_name = "scenario63"

    num_bss = 4
    num_users = 12
    num_tx_ant = 12
    num_rx_ant = 4

    input_path = os.path.join("deepmimo", "raw_scenes")
    output_dir = os.path.join("deepmimo", "channel_outputs")

    # Set the main folder containing extracted scenarios
    parameters = DeepMIMO.default_params()
    parameters['scenario'] = in_name
    parameters['dataset_folder'] = input_path

    parameters['active_BS'] = active_bs
    parameters['user_row_first'] = active_rows[0]
    parameters['user_row_last'] = active_rows[1]

    parameters['num_paths'] = 10

    parameters['bs_antenna']['shape'] = np.array([1, 3, 4])
    parameters['bs_antenna']['spacing'] = 0.5

    parameters['ue_antenna']['shape'] = np.array([1, 2, 2])
    parameters['ue_antenna']['spacing'] = 0.5

    parameters['enable_BS2BS'] = 0

    parameters['OFDM_channels'] = 1  # Frequency (OFDM) or time domain channels
    parameters['OFDM']['subcarriers'] = 1
    parameters['OFDM']['subcarriers_limit'] = 1
    parameters['OFDM']['subcarriers_sampling'] = 1
    parameters['OFDM']['bandwidth'] = 240e-6  # in GHz

    # Generate data
    dataset = DeepMIMO.generate_data(parameters)

    # Choose userindices
    num_user_pos = dataset[0]["user"]["channel"].shape[0]
    # print(dataset[0]["user"]["channel"].shape)

    bs_allowed_idx = []
    bs_num_allowed_idx = []
    num_users_per_bs = num_users // num_bss
    for k in range(num_bss):
        idx = ((np.abs(dataset[k]["user"]["channel"]).sum() > 1e-15) * np.arange(num_user_pos)).astype(int)
        bs_allowed_idx.append(idx)
        bs_num_allowed_idx.append(len(idx))

    training_user_pos_idx = np.zeros([trainingdata_setsize, num_users], dtype=int)
    for i in range(trainingdata_setsize):
        for k in range(num_bss):
            if k in split1:
                st, en = 0, bs_num_allowed_idx[k] // 2
            elif k in split2:
                st, en = bs_num_allowed_idx[k] // 2, None
            else:
                st, en = 0, None
            training_user_pos_idx[i][(k * num_users_per_bs):((k + 1) * num_users_per_bs)] = \
                np.random.choice(bs_allowed_idx[k][st:en:2], num_users_per_bs, replace=False)

    test_user_pos_idx = np.zeros([testdata_setsize, num_users], dtype=int)
    for i in range(testdata_setsize):
        for k in range(num_bss):
            if k in split1:
                st, en = 0, bs_num_allowed_idx[k] // 2
            elif k in split2:
                st, en = bs_num_allowed_idx[k] // 2, None
            else:
                st, en = 0, None
            test_user_pos_idx[i][(k * num_users_per_bs):((k + 1) * num_users_per_bs)] = \
                np.random.choice(bs_allowed_idx[k][(st + 1):en:2], num_users_per_bs, replace=False)

    training_channels = []
    test_channels = []

    for k in range(num_bss):
        training_channels.append([])
        test_channels.append([])
        for i in range(num_users):
            H = np.zeros([trainingdata_setsize, num_rx_ant, num_tx_ant], dtype=complex)
            for s in range(trainingdata_setsize):
                H[s] = dataset[k]["user"]["channel"][training_user_pos_idx[s, i]].squeeze(axis=-1)
            training_channels[k].append(H)

            H = np.zeros([testdata_setsize, num_rx_ant, num_tx_ant], dtype=complex)
            for s in range(testdata_setsize):
                H[s] = dataset[k]["user"]["channel"][test_user_pos_idx[s, i]].squeeze(axis=-1)
            test_channels[k].append(H)

    with open(os.path.join(output_dir, out_name+"_training.pkl"), mode="wb") as fh:
        pickle.dump(training_channels, fh)

    with open(os.path.join(output_dir, out_name + "_test.pkl"), mode="wb") as fh:
        pickle.dump(test_channels, fh)
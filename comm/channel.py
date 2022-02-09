import random
from math import sqrt, pi
import torch
import comm.mathutil as util
import numpy as np
from scipy.io import loadmat


def mimoifc_randcn(bss_dim, users_dim, assignment, batch_size=[], snr=0, weights=1, interference_ratio=1, channel_gain=1, user_noise_pow=1,
                   device=torch.device("cpu")):
    """
    Generates batches of random channel vectors of a MIMO IFC and returns dict per convention.
    :param bss_dim: iterable, contains array sizes of K BSs (equal for all batches)
    :param users_dim: iterable of iterables, which contain array sizes of users (equal for all batches)
    :param assignment: iterable of size len(rx) containing indices assigning each user to a BS (equal for all batches)
    :param snr: iterable or scalar of snr in dB per base station, results in return max powers of 10^(SNR/10), see WMMSE paper, broadcastable to (batch_size, num_bss)
    :param weights: iterable of weights for rates per user or scalar, technically not a channel property, broadcastable to (batch_size, num_users)
    :param interference_ratio: average linear channel gain ratio between direct and interference channels (broadcastable to (batch_size))
    :param device: device to assign tensors to
    :param precision: double or single for float point values

    :return: dictionary, containing
        channels: iterable of K iterables, each containing all channel matrix tensors from the corresponding BS to each user
        bss_assign: iterable of K tensors containing the indices of the assigned users
        users_assign: tensor of size I containing the integers of the assigned BS index for the corresponding user
        tx_pow: size K tensor containing the power budgets for the base stations
        users_noise_pow: tensor of size I containing the noise power of the corresponding user
        rweights: size I tensor containing the rate weights for the individual users
        num_bss: number of basestations
        num_users: number of users
        bss_dim: tensor containing the BS array sizes
        users_dim: tensor containing the user array sizes
        batch_size: list containing batch dimensions
        device: device on which tensors are stored
    """

    dtype = torch.double

    num_bss = len(bss_dim)
    num_users = len(users_dim)
    assert (len(assignment) == len(users_dim))

    # SNR
    bss_pow = torch.pow(torch.tensor(10, device=device, dtype=dtype),
                        torch.tensor(snr, device=device, dtype=dtype) / 10)
    bss_pow = bss_pow.expand([*batch_size, num_bss])

    # Rate weights
    rweights = torch.tensor(weights, device=device, dtype=dtype)
    rweights = rweights.expand([*batch_size, num_users])

    # Noise
    users_noise_pow = torch.full([*batch_size, num_users], user_noise_pow, device=device, dtype=dtype)

    # Channel ratio
    if_channel_element_ratio = torch.tensor(interference_ratio, device=device, dtype=dtype).sqrt()
    if_channel_element_ratio = if_channel_element_ratio.expand([*batch_size]).unsqueeze(-1).unsqueeze(-1)

    channels = []
    bss_assign = []
    users_assign = torch.tensor(assignment, device=device)
    for i_bs in range(num_bss):
        assigned_users = users_assign == i_bs
        channels.append([])
        for i_user in range(num_users):
            channels[i_bs].append(
                (channel_gain**(1/2)) * util.randcn(*batch_size, users_dim[i_user], bss_dim[i_bs], device=device, dtype=dtype) * (
                        assigned_users[i_user] + (~assigned_users[i_user]) * if_channel_element_ratio))
        bss_assign.append((assigned_users * torch.arange(num_users, device=device))[
                              assigned_users])  # gives the indices of the users assigned to the BS

    bss_dim = torch.tensor(bss_dim, device=device)
    users_dim = torch.tensor(users_dim, device=device)

    scenario = {
        "channels": channels,
        "bss_assign": bss_assign,
        "users_assign": users_assign,
        "bss_pow": bss_pow,
        "users_noise_pow": users_noise_pow,
        "rate_weights": rweights,
        "num_bss": num_bss,
        "num_users": num_users,
        "bss_dim": bss_dim,
        "users_dim": users_dim,
        "batch_size": batch_size,
        "device": device,
    }

    return scenario


def mimoifc_triangle(bss_dim, users_dim, assignment, batch_size=[], bss_distance=500, bss_pow=40, noise_pow=-120, weights=1, cell_model="pico", csi_noise=None,
                   device=torch.device("cpu")):
    """
    Generates batches of random channel vectors of a MIMO IFC and returns dict per convention.
    :param bss_dim: iterable, contains array sizes of K BSs (equal for all batches)
    :param users_dim: iterable of iterables, which contain array sizes of users (equal for all batches)
    :param assignment: iterable of size len(rx) containing indices assigning each user to a BS (equal for all batches)
    :param batch_size:
    :param bs_distance: distance of BSs from each other in
    :param bss_pow: combined power budget of BSs, broadcastable to (batch_size, num_bss)
    :param noise_pow: noise power at each antenna of UE, broadcastable to (batch_size, num_users)
    :param weights: iterable of weights for rates per user or scalar, technically not a channel property, broadcastable to (batch_size, num_users)
    :param device: device to assign tensors to
    :param precision: double or single for float point values

    :return: dictionary, containing
        channels: iterable of K iterables, each containing all channel matrix tensors from the corresponding BS to each user
        bss_assign: iterable of K tensors containing the indices of the assigned users
        users_assign: tensor of size I containing the integers of the assigned BS index for the corresponding user
        tx_pow: size K tensor containing the power budgets for the base stations
        users_noise_pow: tensor of size I containing the noise power of the corresponding user
        rweights: size I tensor containing the rate weights for the individual users
        num_bss: number of basestations
        num_users: number of users
        bss_dim: tensor containing the BS array sizes
        users_dim: tensor containing the user array sizes
        batch_size: list containing batch dimensions
        device: device on which tensors are stored
    """

    dtype = torch.double

    num_bss = len(bss_dim)
    num_users = len(users_dim)
    factory_kwargs = {"dtype": dtype, "device": device}

    def get_scene(bs_d):
        def polar_to_cartesian(radius, angle):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            xy = torch.tensor(np.stack([x, y], axis=-1), **factory_kwargs)
            return xy

        tx_pos = [[0, 0]]
        if num_bss >= 2:
            tx_pos.append([bs_d, 0])
        if num_bss >= 3:
            tx_pos.append([bs_d/2, sqrt(3) * bs_d / 2])
        tx_pos = torch.tensor(tx_pos, **factory_kwargs)

        rx_pos = torch.zeros(*batch_size, num_users, 2)
        for i_user in range(num_users):
            R = bs_d / sqrt(3)
            rx_pos_rtemp = R * np.sqrt(np.random.rand(*batch_size) * (1 - 100/(R**2)) + 100/(R**2))  # no closer than 10 m
            rx_pos_phitemp = pi / 3 * np.random.rand(*batch_size)

            i_bs = assignment[i_user]
            if i_bs == 0:
                xyoff = polar_to_cartesian(rx_pos_rtemp, rx_pos_phitemp)
            elif i_bs == 1:
                xyoff = polar_to_cartesian(rx_pos_rtemp, rx_pos_phitemp + (2 * pi / 3))
            elif i_bs == 2:
                xyoff = polar_to_cartesian(rx_pos_rtemp, rx_pos_phitemp + (4 * pi / 3))
            else:
                raise ValueError
            rx_pos[..., i_user, :] = tx_pos[i_bs] + xyoff

        return tx_pos, rx_pos

    def get_pathloss(tx_pos, rx_pos):
        pathloss_mat = torch.zeros(*batch_size, num_users, num_bss, **factory_kwargs)
        for i_bs in range(num_bss):
            for i_user in range(num_users):
                distance = torch.linalg.norm(tx_pos[..., i_bs, :] - rx_pos[..., i_user, :], ord=2, dim=-1)
                if cell_model == "pico":
                    pathloss_mat[..., i_user, i_bs] = 27.9 + 36.7 * torch.log10(distance)  # distance in meters, picocell
                elif cell_model == "macro":
                    pathloss_mat[..., i_user, i_bs] = 15.3 + 37.6 * torch.log10(distance)  # distance in meters, macro cell
                else:
                    raise ValueError("Unknown cell model.")
        return pathloss_mat


    num_bss = len(bss_dim)
    num_users = len(users_dim)
    assert (len(assignment) == len(users_dim))

    # SNR
    bss_pow = torch.pow(torch.tensor(10, **factory_kwargs),
                        torch.tensor(bss_pow, **factory_kwargs) / 10)
    bss_pow = bss_pow.expand(*batch_size, num_bss)

    # Rate weights
    rweights = torch.tensor(weights, **factory_kwargs)
    rweights = rweights.expand([*batch_size, num_users])

    # Noise
    users_noise_pow = torch.pow(torch.tensor(10, **factory_kwargs),
                        torch.tensor(noise_pow, **factory_kwargs) / 10)
    users_noise_pow = users_noise_pow.expand(*batch_size, num_users)

    # Generate scene
    tx_position, rx_position = get_scene(bss_distance)
    pathloss = get_pathloss(tx_position, rx_position)
    # print(pathloss[0])

    channels = []
    bss_assign = []
    users_assign = torch.tensor(assignment, device=device)
    for i_bs in range(num_bss):
        assigned_users = users_assign == i_bs
        channels.append([])
        for i_user in range(num_users):
            channels[i_bs].append(
                (10**(-pathloss[..., i_user, i_bs] / 20)).unsqueeze(-1).unsqueeze(-1)
                * util.randcn(*batch_size, users_dim[i_user], bss_dim[i_bs], **factory_kwargs))  # Rayleigh fading
        bss_assign.append((assigned_users * torch.arange(num_users, device=device))[
                              assigned_users])  # gives the indices of the users assigned to the BS

    bss_dim = torch.tensor(bss_dim, device=device)
    users_dim = torch.tensor(users_dim, device=device)

    scenario = {
        "channels": channels,
        "tx_pos": tx_position,
        "rx_pos": rx_position,
        "pathloss": pathloss,  # in dB, (*batch_size, num_users, num_bss)
        "bss_assign": bss_assign,
        "users_assign": users_assign,
        "bss_pow": bss_pow,
        "users_noise_pow": users_noise_pow,
        "rate_weights": rweights,
        "num_bss": num_bss,
        "num_users": num_users,
        "bss_dim": bss_dim,
        "users_dim": users_dim,
        "batch_size": batch_size,
        "device": device,
    }

    if csi_noise:
        scenario = add_csi_noise(scenario, noise=csi_noise)

    return scenario


def siso_adhoc_2dscene(num_pairs, noise_pow=-92, pathloss_coefficient=2.2, density=1, weights=1, batch_size=[],
                       device=torch.device("cpu"), fixed_scene=False,):
    """
    Generates wireless scenes according to "Eisen and Ribeiro - 2020 -
        Optimal Wireless Resource Allocation With Random Edge Graph Neural Networks".
    :param num_pairs: Number TX-RX pairs.
    :param noise_pow: Rx noise power in dB. If given as 2-tuple with noise_pow[1]>noise_pow[0], then each realization
        has randomly drawn density between noise_pow[0] and noise_pow[1].
    :param path_loss_coeff: Path loss exponent (positive).
    :param density: Divides all distances. If given as 2-tuple with density[1]>density[0], then each realization has
        randomly drawn density between density[0] and density[1].
    :param batch_size: Size of batch realization.
    :param device: Compute device.
    :param precision: "single" or "double".
    :return:
    """
    def large_scale_fading(tx_pos, rx_pos, pathloss_coeff):
        # num_pairs = tx_pos.size()[0]
        large_scale_fading_channel = torch.zeros(*batch_size, num_pairs, num_pairs, **factory_kwargs)
        for i_bs in range(num_pairs):
            for i_user in range(num_pairs):
                large_scale_fading_channel[..., i_user, i_bs] = torch.linalg.norm(tx_pos[..., i_bs, :] - rx_pos[..., i_user, :],
                                                                             ord=2, dim=-1) ** (-pathloss_coeff)

        large_scale_fading_channel = torch.clamp(large_scale_fading_channel, max=1)  # avoid gain
        return large_scale_fading_channel

    def sample_fast_fading(large_scale_fading_chan, sigma=1):
        # num_pairs = large_scale_fading_chan.size()[-1]
        # fast_fading = torch.tensor(fast_fading, dtype=ctype, device=device)
        fast_fading = sigma * util.randcn(*batch_size, num_pairs, num_pairs, dtype=dtype, device=device) * (2**(1/2))  # in order to make sigma consistent in sense of Rayleigh(sig)
        sampled_channel = large_scale_fading_chan * fast_fading
        return sampled_channel

    dtype = torch.double
    ctype = torch.complex128

    factory_kwargs = {"dtype": dtype, "device": device}

    # print(noise_pow)

    # Random decision with num_pairs tuple
    if isinstance(num_pairs, tuple) or isinstance(num_pairs, list):
        num_pairs = random.randrange(*num_pairs)
    else:
        pass

    num_bss = num_pairs
    num_users = num_pairs

    # SNR
    bss_pow = torch.tensor(1, **factory_kwargs)
    bss_pow = bss_pow.expand([*batch_size, num_bss])

    # Rate weights
    rweights = torch.tensor(weights, **factory_kwargs)
    rweights = rweights.expand([*batch_size, num_users])

    # Noise
    if isinstance(noise_pow, tuple) or isinstance(noise_pow, list):
        noise_pow = torch.rand(*batch_size, 1, **factory_kwargs) * (noise_pow[1] - noise_pow[0]) + noise_pow[0]
        users_noise_pow = (10 ** (noise_pow/10)).expand(*batch_size, num_users)
    else:
        users_noise_pow = torch.full([*batch_size, num_users], 10 ** (noise_pow/10), **factory_kwargs)

    # for now fixed positions
    if fixed_scene:
        assert (num_pairs == 20)
        tx_position = torch.tensor([[-2.4502, 10.0507],
                                    [-4.7377, -9.7962],
                                    [10.6207, 0.2383],
                                    [11.8080, 7.9631],
                                    [-12.5251, 15.6361],
                                    [-0.4094, 18.3717],
                                    [-2.1766, 1.8886],
                                    [5.8525, -14.4550],
                                    [8.3746, -14.0282],
                                    [10.1875, -9.6997],
                                    [-8.9590, 13.6287],
                                    [7.1881, -9.8287],
                                    [6.2039, 12.5714],
                                    [-13.4955, -10.2590],
                                    [-15.2401, 17.1705],
                                    [-0.0654, -6.0006],
                                    [18.3898, -12.1362],
                                    [-6.3846, -9.9566],
                                    [3.4107, 4.6418],
                                    [-11.0475, -1.0684]], **factory_kwargs)

        rx_position = torch.tensor([[1.0801, 12.8532],
                                    [-3.5171, -10.8988],
                                    [9.1302, -2.3448],
                                    [11.9405, 7.0022],
                                    [-13.5070, 11.6007],
                                    [-4.6498, 14.6914],
                                    [-4.7774, 6.3091],
                                    [2.0857, -9.8937],
                                    [5.2137, -13.2762],
                                    [7.5870, -14.1019],
                                    [-9.7863, 10.9765],
                                    [2.6847, -11.2971],
                                    [10.2311, 15.7833],
                                    [-9.0477, -15.1050],
                                    [-15.3315, 12.6008],
                                    [-0.1729, -9.3107],
                                    [16.7670, -10.6450],
                                    [-2.3840, -7.6394],
                                    [2.1032, 6.1192],
                                    [-14.9355, -1.5592],
                                    ], **factory_kwargs)
        # tx_position = tx_position.expand(*batch_size, num_pairs, 2)
        # rx_position = rx_position.expand(*batch_size, num_pairs, 2)
    else:
        if isinstance(density, tuple) or isinstance(density, list):
            density = torch.rand(*batch_size, 1, 1, **factory_kwargs) * (density[1] - density[0]) + density[0]

        tx_position = (2 * num_users * torch.rand(*batch_size, num_pairs, 2, **factory_kwargs) - num_users) / density
        rx_position = tx_position + ((num_users / 2) * torch.rand(*batch_size, num_pairs, 2, **factory_kwargs) - num_users/4)  # [-K/4, K/4]

    users_assign = torch.arange(num_pairs, device=device)
    channels = []
    bss_assign = []
    large_scale_fading_mat = large_scale_fading(tx_position, rx_position, pathloss_coefficient)
    sampled_channel_mat = sample_fast_fading(large_scale_fading_mat)
    for i_bs in range(num_bss):
        assigned_users = users_assign == i_bs
        channels.append([])
        for i_user in range(num_users):
            channels[i_bs].append(sampled_channel_mat[..., i_user, i_bs].unsqueeze(-1).unsqueeze(-1))
        bss_assign.append((assigned_users * torch.arange(num_users, device=device))[
                              assigned_users])  # gives the indices of the users assigned to the BS

    bss_dim = torch.tensor([1] * num_bss, device=device)
    users_dim = torch.tensor([1] * num_bss, device=device)
    scenario = {
        "tx_pos": tx_position,
        "rx_pos": rx_position,
        "channels": channels,
        "pathloss": large_scale_fading_mat,
        "bss_assign": bss_assign,
        "users_assign": users_assign,
        "bss_pow": bss_pow,
        "users_noise_pow": users_noise_pow,
        "rate_weights": rweights,
        "num_bss": num_bss,
        "num_users": num_users,
        "bss_dim": bss_dim,
        "users_dim": users_dim,
        "batch_size": batch_size,
        "device": device,
    }
    # print(channels[0][0].size())
    return scenario


def scenario_select_index(scenario, ind):
    """
    Returns a scenario with only the values corresponding to the scenario index of the original batch.
    :param scenario: mimo_8Tx ifc scenario
    :param index: index or indices within batch as iterable or iterable of iterables with length equal to the batch ndim
    :return: Reduced scenario.
    """

    def nested_select_index(item, ind):
        """
        Crawsl through nested lists and returns a new nested list with reduced tensors.
        :param item: tensor or nested list of tensors
        :param item: index or indices within batch as list or list of lists with length equal to the batch ndim
        :return: reduced tensor or nested list of reduced tensors
        """

        def list_index_select(tensor, ind):
            selected_tensor = tensor
            if hasattr(ind[0], "__len__"):
                for nested_ind in ind:
                    selected_tensor = selected_tensor[nested_ind]
            else:
                selected_tensor = selected_tensor[ind]
            return selected_tensor

        if isinstance(item, list):
            reduced_item = []
            for nested_item in item:
                reduced_item.append(nested_select_index(nested_item, ind))
        elif torch.is_tensor(item):
            reduced_item = list_index_select(item, ind)

        return reduced_item

    batched_data_dim = {"channels": 2,
                        "noisy_channels": 2,
                        "bss_pow": 1,
                        "users_noise_pow": 1,
                        "rate_weights": 1}
    if "batch_size" in scenario:
        batch_size = scenario["batch_size"]
    else:
        batch_size = list(scenario["rate_weights"].size())[:-1]  # legacy
    batch_ndim = len(batch_size)
    assert (len(ind) == batch_ndim)

    reduced_scenario = {}
    for k, v in scenario.items():
        if k in batched_data_dim.keys():
            reduced_scenario[k] = nested_select_index(v, ind)
        else:
            reduced_scenario[k] = v

    return reduced_scenario


def add_csi_noise(scenario, noise):
    """
    Generate a scenario with added noise.
    :param noise: in dB, additive to channel matrix elements.
    """
    device = scenario["device"]
    rtype = scenario["bss_pow"].dtype
    linear_noise_pow = torch.tensor(10**(noise / 10), dtype=rtype)

    noisy_scenario = {}
    for k, v in scenario.items():
        noisy_scenario[k] = v

    v = scenario["channels"]
    noisy_channels = []
    for i_bs in range(len(v)):
        noisy_channels.append([])
        for i_user in range(len(v[i_bs])):
            ch_size = list(v[i_bs][i_user].size())
            noisy_channels[i_bs].append(v[i_bs][i_user] + torch.sqrt(linear_noise_pow) * util.randc(*ch_size, dtype=rtype, device=device))
    noisy_scenario["noisy_channels"] = noisy_channels

    return noisy_scenario


def prepare_csi(scenario):
    """Replaces channels with noisy_channels if available."""
    if "noisy_channels" in scenario:
        noisy_scenario = {}
        for k, v in scenario.items():
            if k == "channels":
                pass
            if k == "noisy_channels":
                noisy_scenario["channels"] = v
            else:
                noisy_scenario[k] = v
        return noisy_scenario
    else:
        return scenario


def downlink_rate(scenario, dl_beamformer):
    """
    Computes the rate and weighted rate per user.
    :param scenario: mimo_8Tx ifc scenario, or batched scenario of same batch_isze as dl_beamformer
    :param dl_beamformer: iterable of (*, M, D) tensors, where the matrix dim DxM can differ from tensor to tensor, but the batch dim * cannot.
    :return: rate, weighted rate
    """
    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    rweights = scenario["rate_weights"]

    batch_size = list(dl_beamformer[0].size())[:-2]
    batch_ndim = len(batch_size)
    expand_dim = [1] * batch_ndim

    bf_stacks = []
    for i_bs in range(num_bss):
        bs_assigned_users = scenario["bss_assign"][i_bs]  # gets all users for current BS
        bf_stacks.append(torch.stack([dl_beamformer[i_user] for i_user in bs_assigned_users],
                                     dim=0))  # builds tensor from stacking all users of bs in dim 0, only works if stackable actually

    dl_rate = []
    for i_user in range(num_users):
        user_dim = scenario["users_dim"][i_user]
        # user_assigned_bs = scenario["users_assign"][i_user]
        eye = torch.eye(user_dim, device=device).view(*expand_dim, user_dim, user_dim)

        noise_sum_mat = eye * scenario["users_noise_pow"][..., i_user].unsqueeze(-1).unsqueeze(-1)
        user_pow_mat = torch.zeros(*batch_size, user_dim, user_dim, device=device)
        for i_bs in range(num_bss):
            bs_dim = scenario["bss_dim"][i_bs]
            bs_assigned_users = scenario["bss_assign"][i_bs]  # gets all users for current BS
            interfering_user_mask = bs_assigned_users != i_user  # all interfering BFs of BS for current user
            pos_interference_user = torch.arange(len(interfering_user_mask), device=device).masked_select(
                interfering_user_mask)  # workaround until masked select works
            bs_channel2user = scenario["channels"][i_bs][i_user].expand(
                [1, *batch_size, user_dim, bs_dim])  # unsqueezes into same dimensions as bf_stack
            pow_mats = torch.matmul(bs_channel2user, bf_stacks[i_bs])
            pow_mats = util.cmat_square(pow_mats)

            if ~torch.all(interfering_user_mask):
                i_pos_user = torch.arange(len(interfering_user_mask), device=device).masked_select(
                    ~interfering_user_mask)  # workaround until masked select works
                user_pow_mat = pow_mats.index_select(0, i_pos_user).squeeze(
                    dim=0)  # index select does not remove indexed dim

            noise_sum_mat = noise_sum_mat + pow_mats.index_select(0, pos_interference_user).sum(dim=0)

        current_sinr = torch.matmul(user_pow_mat, util.clean_hermitian(torch.linalg.inv(noise_sum_mat)))
        current_rate = util.logdet(eye + current_sinr)

        dl_rate.append(current_rate)

    dl_rate = torch.stack(dl_rate, dim=-1)  # users in last dim
    weighted_dl_rate = dl_rate * rweights.expand([*batch_size, num_users])

    return dl_rate, weighted_dl_rate


def downlink_sum_rate(scenario, dl_beamformer):
    """
    Computes the sum-rate and weighted sum-rate per user.
    :param scenario: mimo_8Tx ifc scenario
    :param tx_beamformer: iterable of (*, M, D) tensors, where the matrix dim DxM can differ from tensor to tensor, but the batch dim * cannot.
    :return: sum-rate, weighted sum-rate
    """
    rate, wrate = downlink_rate(scenario, dl_beamformer)
    sum_rate, weighted_sum_rate = rate.sum(dim=-1), wrate.sum(dim=-1)

    return sum_rate, weighted_sum_rate


def downlink_rate_iaidnn(scenario, dl_beamformer):
    """
    Computes the IAIDNN objective (normalizes the power of the beamformers after the fact).
    :param scenario: mimo_8Tx ifc scenario, or batched scenario of same batch_isze as dl_beamformer
    :param dl_beamformer: iterable of (*, M, D) tensors, where the matrix dim DxM can differ from tensor to tensor, but the batch dim * cannot.
    :return: rate, weighted rate
    """
    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    rweights = scenario["rate_weights"]
    bss_max_pow = scenario["bss_pow"]
    assigned_bs = scenario["users_assign"]

    batch_size = list(dl_beamformer[0].size())[:-2]
    batch_ndim = len(batch_size)
    expand_dim = [1] * batch_ndim
    dtype = bss_max_pow.dtype

    bf_stacks = []
    for i_bs in range(num_bss):
        bs_assigned_users = scenario["bss_assign"][i_bs]  # gets all users for current BS
        bf_stacks.append(torch.stack([dl_beamformer[i_user] for i_user in bs_assigned_users],
                                     dim=0))  # builds tensor from stacking all users of bs in dim 0

    bss_actual_pow = torch.zeros(*batch_size, num_bss, dtype=dtype)
    for i_user in range(num_users):
        user_bs = assigned_bs[i_user]
        bss_actual_pow[..., user_bs] = bss_actual_pow[..., user_bs] + util.bf_mat_pow(dl_beamformer[i_user])

    dl_rate = []
    for i_user in range(num_users):
        user_dim = scenario["users_dim"][i_user]
        user_assigned_bs = scenario["users_assign"][i_user]
        eye = torch.eye(user_dim, device=device).view(*expand_dim, user_dim, user_dim)

        noise_sum_mat = eye * scenario["users_noise_pow"][..., i_user].unsqueeze(-1).unsqueeze(-1) * \
                        bss_actual_pow[..., user_assigned_bs].unsqueeze(-1).unsqueeze(-1) / \
                        bss_max_pow[..., user_assigned_bs].unsqueeze(-1).unsqueeze(-1)
        user_pow_mat = torch.zeros(*batch_size, user_dim, user_dim, device=device)
        for i_bs in range(num_bss):
            bs_dim = scenario["bss_dim"][i_bs]
            bs_assigned_users = scenario["bss_assign"][i_bs]  # gets all users for current BS
            interfering_user_mask = bs_assigned_users != i_user  # all interfering BFs of BS for current user
            pos_interference_user = torch.arange(len(interfering_user_mask), device=device).masked_select(
                interfering_user_mask)  # workaround until masked select works
            bs_channel2user = scenario["channels"][i_bs][i_user].expand(
                [1, *batch_size, user_dim, bs_dim])  # unsqueezes into same dimensions as bf_stack
            pow_mats = torch.matmul(bs_channel2user, bf_stacks[i_bs])
            pow_mats = util.cmat_square(pow_mats)

            if ~torch.all(interfering_user_mask):
                i_pos_user = torch.arange(len(interfering_user_mask), device=device).masked_select(
                    ~interfering_user_mask)  # workaround until masked select works
                user_pow_mat = pow_mats.index_select(0, i_pos_user).squeeze(
                    dim=0)  # index select does not remove indexed dim

            noise_sum_mat = noise_sum_mat + pow_mats.index_select(0, pos_interference_user).sum(dim=0)

        current_sinr = torch.matmul(user_pow_mat, util.clean_hermitian(torch.linalg.inv(noise_sum_mat)))
        current_rate = util.logdet(eye + current_sinr)
        dl_rate.append(current_rate)

    dl_rate = torch.stack(dl_rate, dim=-1)  # users in last dim
    weighted_dl_rate = dl_rate * rweights.expand([*batch_size, num_users])

    return dl_rate, weighted_dl_rate


def downlink_sum_rate_iaidnn(scenario, dl_beamformer):
    """
    Computes the sum-rate and weighted sum-rate per user.
    :param scenario: mimo_8Tx ifc scenario
    :param tx_beamformer: iterable of (*, M, D) tensors, where the matrix dim DxM can differ from tensor to tensor, but the batch dim * cannot.
    :return: sum-rate, weighted sum-rate
    """
    rate, wrate = downlink_rate_iaidnn(scenario, dl_beamformer)
    sum_rate, weighted_sum_rate = rate.sum(dim=-1), wrate.sum(dim=-1)
    # print(sum_rate[-1][0])
    return sum_rate, weighted_sum_rate


def deepmimo(channel_path=None, bss_pow=30, noise_pow=-90, weights=1, precision="double", device=torch.device("cpu")):
    """
    Generates a scenario batch using the DeepMIMO scenario batch in the Matlab .mat file under channel_path.
    :return: dictionary, containing
        channels: iterable of K iterables, each containing all channel matrix tensors from the corresponding BS to each user
        bss_assign: iterable of K tensors containing the indices of the assigned users
        users_assign: tensor of size I containing the integers of the assigned BS index for the corresponding user
        tx_pow: size K tensor containing the power budgets for the base stations
        users_noise_pow: tensor of size I containing the noise power of the corresponding user
        rweights: size I tensor containing the rate weights for the individual users
        num_bss: number of basestations
        num_users: number of users
        bss_dim: tensor containing the BS array sizes
        users_dim: tensor containing the user array sizes
        batch_size: list containing batch dimensions
        device: device on which tensors are stored
    """

    if precision == "double":
        dtype = torch.double
    else:
        dtype = torch.single
    factory_kwargs = {"dtype": dtype, "device": device}

    # Data extraction
    matfile = loadmat(channel_path)
    channel_mats = matfile["channels"]
    num_bss = channel_mats.shape[1]
    txdim = channel_mats[0, 0][0, 0].shape[2]
    num_users = channel_mats[0, 0].shape[1]
    rxdim = channel_mats[0, 0][0, 0].shape[1]
    batch_size = [channel_mats[0, 0][0, 0].shape[0]]
    # print(num_bss, num_users, batch_size)

    bss_dim = [*[txdim]*num_bss]
    users_dim = [*[rxdim]*num_users]
    assignment = []
    for ibs in range(num_bss):
        assignment.extend([*[ibs] * int(num_users/num_bss)])

    # batch_size =
    num_bss = len(bss_dim)
    num_users = len(users_dim)
    assert (len(assignment) == len(users_dim))

    # SNR
    bss_pow = torch.pow(torch.tensor(10, **factory_kwargs),
                        torch.tensor(bss_pow, **factory_kwargs) / 10)
    bss_pow = bss_pow.expand(*batch_size, num_bss)

    # Rate weights
    rweights = torch.tensor(weights, **factory_kwargs)
    rweights = rweights.expand([*batch_size, num_users])

    # Noise
    users_noise_pow = torch.pow(torch.tensor(10, **factory_kwargs),
                        torch.tensor(noise_pow, **factory_kwargs) / 10)
    users_noise_pow = users_noise_pow.expand(*batch_size, num_users)

    channels = []
    bss_assign = []
    users_assign = torch.tensor(assignment, device=device)
    for i_bs in range(num_bss):
        assigned_users = users_assign == i_bs
        channels.append([])
        for i_user in range(num_users):
            channels[i_bs].append(torch.tensor(channel_mats[0, i_bs][0, i_user]))
        bss_assign.append((assigned_users * torch.arange(num_users, device=device))[
                              assigned_users])  # gives the indices of the users assigned to the BS

    bss_dim = torch.tensor(bss_dim, device=device)
    users_dim = torch.tensor(users_dim, device=device)

    scenario = {
        "channels": channels,
        "bss_assign": bss_assign,
        "users_assign": users_assign,
        "bss_pow": bss_pow,
        "users_noise_pow": users_noise_pow,
        "rate_weights": rweights,
        "num_bss": num_bss,
        "num_users": num_users,
        "bss_dim": bss_dim,
        "users_dim": users_dim,
        "batch_size": batch_size,
        "path": channel_path,
        "device": device,
    }

    return scenario

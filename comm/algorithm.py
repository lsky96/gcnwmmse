"""
For WMMSE, see Shi et al. - 2011 - An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel
"""

import math
import torch
import comm.mathutil as util
import comm.channel as channel


def downlink_mrc(scenario):
    """
    Calculates the equally weighted MRC beamformers for the scenario which fulfill the power constraint
    :param scenario: mimo_8Tx ifc scenario
    :return: iterable of K beamformers for each user k
    """
    num_users = scenario["num_users"]
    users_assign = scenario["users_assign"]
    channels = scenario["channels"]
    bss_maxpow = scenario["bss_pow"]

    dl_beamformers = []
    bss_pow = torch.zeros_like(bss_maxpow)  # sums up power of BS for later correction

    for i_user in range(num_users):
        i_bs = users_assign[i_user]
        assigned_beamformer = channels[i_bs][i_user].conj().transpose(-2, -1)
        dl_beamformers.append(assigned_beamformer)
        bss_pow[..., i_bs] += util.bf_mat_pow(assigned_beamformer)  # needs to be casted

    for i_user in range(num_users):
        i_bs = users_assign[i_user]
        dl_beamformers[i_user] = dl_beamformers[i_user] * torch.sqrt(bss_maxpow[..., i_bs] / bss_pow[..., i_bs]).unsqueeze(-1).unsqueeze(-1)

    return dl_beamformers


def downlink_zf_bf(scenario, mode="standard"):
    """
    Naive ZF beamformer, inverse potentially unstable. If mode is "iaidnn", power will not be normalized, as in IAIDNN reference code.
    """
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    bss_assign = scenario["bss_assign"]
    users_assign = scenario["users_assign"]
    bss_maxpow = scenario["bss_pow"]
    channels = scenario["channels"]

    inv_ul_cov = []
    for i_bs in range(num_bss):
        inv_ul_cov_temp = 0
        for i_user in bss_assign[i_bs]:
            inv_ul_cov_temp = inv_ul_cov_temp + util.cmat_square(channels[i_bs][i_user].conj().transpose(-2, -1))
        inv_ul_cov.append(inv_ul_cov_temp)

    dl_beamformers = []
    bss_pow = torch.zeros_like(bss_maxpow)  # sums up power of BS for later correction
    for i_user in range(num_users):
        i_bs = users_assign[i_user]
        assigned_beamformer = torch.matmul(inv_ul_cov[i_bs], channels[i_bs][i_user].conj().transpose(-2, -1))
        bss_pow[..., i_bs] += util.bf_mat_pow(assigned_beamformer)
        dl_beamformers.append(assigned_beamformer)

    if not mode == "iaidnn":
        for i_user in range(num_users):
            i_bs = users_assign[i_user]
            dl_beamformers[i_user] = dl_beamformers[i_user] * torch.sqrt(bss_maxpow[..., i_bs] / bss_pow[..., i_bs]).unsqueeze(-1).unsqueeze(-1)

    return dl_beamformers


def downlink_randn_bf(scenario, bf_samples=0):
    """
    Constructs CN(0,1) random downlink beamformers normalized to the power requirements
    :param scenario: DL mimo_8Tx ifc scenario
    :param bf_samples: number of generated beamformers per scenario sample. In case of 0, still one sample, but without additional batch dim.
    :return:
    """
    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    bss_assign = scenario["bss_assign"]
    users_assign = scenario["users_assign"]
    bss_dim = scenario["bss_dim"]
    users_dim = scenario["users_dim"]
    bss_maxpow = scenario["bss_pow"]
    dtype = bss_maxpow.dtype
    orig_batch_size = list(bss_maxpow.size())[:-1]
    if bf_samples >= 1:
        batch_size = [bf_samples] + orig_batch_size
    else:
        batch_size = orig_batch_size

    dl_beamformers = []
    bss_pow = torch.zeros(*batch_size, num_bss, device=device, dtype=dtype)  # sums up power of BS for later correction

    for i_user in range(num_users):
        i_bs = users_assign[i_user]
        assigned_beamformer = util.randcn(*batch_size, bss_dim[i_bs], users_dim[i_user], device=device, dtype=dtype)
        dl_beamformers.append(assigned_beamformer)
        bss_pow[..., i_bs] += util.bf_mat_pow(assigned_beamformer)  # needs to be casted

    for i_user in range(num_users):
        i_bs = users_assign[i_user]
        num_users_assigned = len(bss_assign[i_bs])
        if num_users_assigned == 1:
            dl_beamformers[i_user] = dl_beamformers[i_user] * \
                                     torch.sqrt(torch.rand(*batch_size, device=device) * bss_maxpow[..., i_bs] / bss_pow[..., i_bs]).unsqueeze(-1).unsqueeze(-1)
        else:
            dl_beamformers[i_user] = dl_beamformers[i_user] * torch.sqrt(bss_maxpow[..., i_bs] / bss_pow[..., i_bs]).unsqueeze(-1).unsqueeze(-1)

    return dl_beamformers


def downlink_wmmse(scenario, init_dl_beamformer=[], num_iter=50, show_iteration=True):
    """
    :param scenario:
    :param init_dl_beamformer: iterable of matrices, initial guess for the downlink beamformer, matrices are in batch-form
    :param num_iter: number of iterations
    :return:
    """
    debug = False
    num_layer = num_iter

    # Internal configs
    lagrangian_max_iter = 16

    if not init_dl_beamformer:
        init_dl_beamformer = downlink_randn_bf(scenario)

    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    assigned_bs = scenario["users_assign"]
    rweights = scenario["rate_weights"]

    bss_dim = scenario["bss_dim"]
    users_dim = scenario["users_dim"]
    bss_maxpow = scenario["bss_pow"]

    batch_size = list(init_dl_beamformer[0].size())[0:-2]
    batch_ndim = init_dl_beamformer[0].ndim - 2  # num dimension before matrix dimensions
    expand_dim = [1] * batch_ndim

    rtype = scenario["bss_pow"].dtype

    # prepare and expand channels
    channels = []
    for i_bs in range(num_bss):
        channels.append([])
        for i_user in range(num_users):
            channels[i_bs].append(scenario["channels"][i_bs][i_user].expand([*batch_size, *list(scenario["channels"][i_bs][i_user].size())[-2:]]))  # extends channels into batch dim

    # layer or iteration dimension will be added as first dimension for all tensors.
    v, u, w = [], [], []
    for i_user in range(num_users):
        v.append([])
        u.append([])
        w.append([])
        v[i_user].append(init_dl_beamformer[i_user])

    for i_layer in range(num_layer):
        if debug or show_iteration:
            print("\rIteration {}...".format(i_layer+1), end="", flush=True)

        # MMSE: U + W for every user
        # U * W * U^H is equal for every BS, it will be calculated concurrently as preparation for calculation of V
        uwu = []
        v_tilde = []  # preps for calc of V
        for i_user in range(num_users):
            # SUM(H * V * V^H * H) + sigma * I
            user_dim = scenario["users_dim"][i_user]
            eye = torch.eye(user_dim, device=device).view(*expand_dim, user_dim, user_dim)
            dl_covariance_mat = scenario["users_noise_pow"][..., i_user].unsqueeze(-1).unsqueeze(-1) * eye

            # SUM efficiently calculated
            for i_bs in range(num_bss):
                bs_assigned_users = scenario["bss_assign"][i_bs]
                v_stack = torch.stack([v[lj][i_layer] for lj in bs_assigned_users], dim=0)
                bs_channel2user = channels[i_bs][i_user]
                partial_covariance_mats = torch.matmul(bs_channel2user, v_stack)
                partial_covariance_mats = util.cmat_square(partial_covariance_mats)
                dl_covariance_mat = dl_covariance_mat + partial_covariance_mats.sum(dim=0)

            # U = INV * H * V
            inv_cov_mat = util.clean_hermitian(torch.linalg.inv(util.clean_hermitian(dl_covariance_mat)))  # second clean did not help
            user_channel = channels[assigned_bs[i_user]][i_user]
            u_temp = torch.matmul(inv_cov_mat, user_channel)  # takes channel mat of correct BS
            u_temp = torch.matmul(u_temp, v[i_user][i_layer])
            u[i_user].append(u_temp)

            # W = INV(I - U^H * H * V)
            w_inv = torch.matmul(u_temp.conj().transpose(-2, -1), user_channel)
            w_inv = util.clean_hermitian(torch.matmul(w_inv, v[i_user][i_layer]))
            w_inv = eye - w_inv
            w_temp = util.clean_hermitian(torch.linalg.inv(w_inv))
            w[i_user].append(w_temp)

            # U * W * U^H prepares for calc of V
            uwu_temp = torch.matmul(u_temp, w_temp)
            uwu_temp = util.clean_hermitian(torch.matmul(uwu_temp, u_temp.conj().transpose(-2, -1)))  # matrix must be hermitian
            uwu.append(uwu_temp)

            # Preperation V_tilde
            v_tilde_temp = util.mmchain(user_channel.conj().transpose(-2, -1), u_temp, w_temp)
            v_tilde_temp = v_tilde_temp * rweights[..., i_user].unsqueeze(-1).unsqueeze(-1)
            v_tilde.append(v_tilde_temp)

        "CALCULATION OF V"
        inv_augmented_ul_cov_mats_scaled = []
        v_tilde_scaled = [None] * num_users

        for i_bs in range(num_bss):
            partial_ul_cov_mats = []
            # Calculation of UL Covariance SUM
            for i_user in range(num_users):
                bs_channel2user = channels[i_bs][i_user]
                # alpha * H^H * U * W * U^H * H
                partial_ul_cov_mat_temp = bs_channel2user.conj().transpose(-2, -1) @ uwu[i_user]
                partial_ul_cov_mat_temp = partial_ul_cov_mat_temp @ bs_channel2user

                partial_ul_cov_mat_temp = util.clean_hermitian(partial_ul_cov_mat_temp) * rweights[..., i_user].unsqueeze(-1).unsqueeze(-1)
                partial_ul_cov_mats.append(partial_ul_cov_mat_temp)

            # Scaling of cov mat
            ul_cov_mat_temp = torch.stack(partial_ul_cov_mats, dim=0).sum(dim=0)  # sums all partial mats
            scaling_factor = torch.tensor(1, dtype=rtype)  # util.btrace(ul_cov_mat_temp) / bss_dim[i_bs]  # torch.tensor(1, dtype=rtype)
            ul_cov_mat_scaled_temp = ul_cov_mat_temp / scaling_factor.unsqueeze(-1).unsqueeze(-1)

            # assigned usersauxiliary matrix SUM(HUWWUH)
            aux_assigned_user_mat = []
            for i_user in scenario["bss_assign"][i_bs]:  # every user assigned to i_bs
                v_tilde_scaled_temp = v_tilde[i_user] / scaling_factor.unsqueeze(-1).unsqueeze(-1)
                v_tilde_scaled[i_user] = v_tilde_scaled_temp
                aux_assigned_user_mat_temp = util.cmat_square(v_tilde_scaled_temp)
                aux_assigned_user_mat.append(aux_assigned_user_mat_temp)
            aux_assigned_user_mat = torch.stack(aux_assigned_user_mat, dim=0).sum(dim=0)

            # Decomposition of SUM for Lagrangian
            eigenval_temp, eigenvec_temp = torch.linalg.eigh(ul_cov_mat_temp)
            eigenval_temp = torch.clamp(eigenval_temp, min=0)  # required for stability
            eigenval_temp = eigenval_temp.movedim(-1, 0)  # .float()  # now dim 0 contains M eigenval, so (M, *batch_size, 1)
            eigenvec_temp = eigenvec_temp.unsqueeze(0).transpose(0, -1)  # .type(torch.complex64)  # the matrix dim now only hold single eigenvecs, while dim 0 now iterates over different eigenvecs, so (M, *batchdim, M, 1)
            nominator_coeff = util.mmchain(eigenvec_temp.conj().transpose(-2, -1), aux_assigned_user_mat.unsqueeze(0), eigenvec_temp).real.squeeze(-1).squeeze(-1)  # dim 0 of size M holds coefficients, so (M, *batch_size, 1)

            nominator_coeff = nominator_coeff / bss_maxpow[..., i_bs].expand(1, *batch_size)
            nominator_coeff = torch.clamp(nominator_coeff, min=0)  # required for stability

            mu_root = util.rationalfct_solve_0d2(nominator_coeff, eigenval_temp, num_iter=lagrangian_max_iter)

            # Compute inverse mat
            eye = torch.eye(bss_dim[i_bs], device=device).view(*expand_dim, bss_dim[i_bs], bss_dim[i_bs])
            inv_augmented_ul_cov_mats_scaled.append(util.clean_hermitian(torch.linalg.inv(ul_cov_mat_scaled_temp + eye * mu_root.unsqueeze(-1).unsqueeze(-1))))

        # Calculation of V
        bss_current_pow = torch.zeros(*batch_size, num_bss, device=device)
        for i_user in range(num_users):
            i_bs = assigned_bs[i_user]

            # Update of V
            v_temp = inv_augmented_ul_cov_mats_scaled[i_bs] @ v_tilde_scaled[i_user]
            v[i_user].append(v_temp)
            bss_current_pow[..., i_bs] += util.bf_mat_pow(v_temp)

        # Power clamping in case of ill conditioned matrices
        bss_current_pow = torch.maximum(bss_maxpow, bss_current_pow)  # pre clamping, to avoid any division underflows
        correction_factor = torch.clamp(torch.sqrt(bss_maxpow / bss_current_pow), max=1)

        for i_user in range(num_users):
            i_bs = assigned_bs[i_user]
            v[i_user][i_layer+1] = v[i_user][i_layer+1] * correction_factor[..., i_bs].unsqueeze(-1).unsqueeze(-1)

    # Stack matrices and check power
    bss_pow = torch.zeros(num_bss, num_layer+1, *batch_size, device=device)
    for i_user in range(num_users):
        v[i_user] = torch.stack(v[i_user], dim=0)
        u[i_user] = torch.stack(u[i_user], dim=0)
        w[i_user] = torch.stack(w[i_user], dim=0)

        i_bs = assigned_bs[i_user]
        bss_pow[i_bs] += util.bf_mat_pow(v[i_user])
    bss_pow = bss_pow.movedim(0, -1)  # moves bss dimension to last, first dim is over layers

    if debug or show_iteration:
        print("done!")

    return v, u, w, bss_pow


def downlink_wmmse50(scenario, num_iter=100, num_trials=100, use_stable=True):
    """
    Performs WMMSE 100 with random initialization to find a semi-optimal rate and the average achieved rate by the WMMSE.
    :param scenario:
    :param init_dl_beamformer:
    :param max_iter:
    :return: wrates_layers (layers, *batch_size), bsspow_layers (layers, *batch_size, num_users)
    """

    # segment due to memory
    batch_size = list(scenario["channels"][0][0].size())[:-2]  # scenario["batch_size"]
    bss_dim = scenario["bss_dim"][0].item()
    # print((math.prod(batch_size) * num_iter))
    num_trials_atonce = int(100000 // (math.prod(batch_size) * num_iter * ((bss_dim**2) / 32)))  # heuristic, should actually be memory dependent
    num_init_batches = math.ceil(num_trials / num_trials_atonce)

    wrates_trials = []
    bss_pow_trials = []
    trials_computed = 0
    i_batch = 1
    while trials_computed < num_trials:
        print("Init batch {} of {}:".format(i_batch, num_init_batches))
        run_num_trials = min(num_trials_atonce, num_trials - trials_computed)
        # print(run_num_trials)
        init_dl_beamformer = downlink_randn_bf(scenario, bf_samples=run_num_trials)
        # print(init_dl_beamformer[0].size())
        csi = channel.prepare_csi(scenario)
        if not scenario_is_siso_adhoc(scenario):
            if use_stable:
                dl_beamformer_trials_temp, _, _, bss_pow_trials_temp = \
                    downlink_wmmse_stable(csi, init_dl_beamformer=init_dl_beamformer, num_iter=num_iter, show_iteration=True)
            else:
                dl_beamformer_trials_temp, _, _, bss_pow_trials_temp = \
                    downlink_wmmse(csi, init_dl_beamformer=init_dl_beamformer, num_iter=num_iter, show_iteration=True)
        else:
            dl_beamformer_trials_temp, _, _, bss_pow_trials_temp = \
                downlink_wmmse_sisoadhoc(csi, init_dl_beamformer=init_dl_beamformer, num_iter=num_iter, show_iteration=True)

        # print(dl_beamformer_trials_temp[0].size())
        # print(bss_pow_trials_temp.size())
        _, wrates_trials_temp = channel.downlink_sum_rate(scenario, dl_beamformer_trials_temp)
        # print(wrates_trials_temp.size())

        wrates_trials.append(wrates_trials_temp)
        bss_pow_trials.append(bss_pow_trials_temp)
        # print(wrates_trials_temp.size())
        # print(bss_pow_trials_temp.size())
        trials_computed += run_num_trials
        i_batch += 1

    wrates_trials = torch.cat(wrates_trials, dim=1)  # cat in trial dim
    bss_pow_trials = torch.cat(bss_pow_trials, dim=1)  # cat in trial dim
    num_layer = list(wrates_trials.size())[0]
    _, index_opt = torch.max(wrates_trials[-1].view(num_trials, -1), dim=0)  # last layer, best over inits, flatten batch

    wrates_trial_avg_layers = torch.mean(wrates_trials, dim=1)
    bss_pow_trial_avg_layers = torch.mean(bss_pow_trials, dim=1)
    bss_pow_ratio_trial_avg_layers = bss_pow_trial_avg_layers / scenario["bss_pow"].unsqueeze(0)

    # flatten batch dim, then pick optimal index and corresponding batch, and reshape again to batch dim
    wrates_best_layers = wrates_trials.view(num_layer, num_trials, index_opt.numel())[:, index_opt, torch.arange(index_opt.numel())].view(num_layer, *batch_size)
    bss_pow_best_layers = bss_pow_trials.view(num_layer, num_trials, index_opt.numel(), -1)[:, index_opt, torch.arange(index_opt.numel())].view(num_layer, *batch_size, -1)
    bss_pow_ratio_best_layers = bss_pow_best_layers / scenario["bss_pow"].unsqueeze(0)

    return wrates_trial_avg_layers, wrates_best_layers, \
           bss_pow_trial_avg_layers, bss_pow_ratio_trial_avg_layers, \
           bss_pow_best_layers, bss_pow_ratio_best_layers


def large_scenario_set_downlink_wmmse50(scenario, num_iter=100, num_trials=100, use_stable=True):
    """
    Automatically splits up a scenario batch and passes it to WMMSE50 due to memory constraints.
    """
    num_samples = scenario["channels"][0][0].size(0)  # scenario["batch_size"]
    bss_dim = scenario["bss_dim"][0].item()
    num_trials_atonce = 100000 / (num_samples * num_iter * ((bss_dim**2) / 32))  # rough estimate to manage memory

    num_samples_atonce = math.floor(num_samples * num_trials_atonce)
    num_subsets = math.ceil(num_samples / num_samples_atonce)
    wrates_trial_avg_layers, wrates_best_layers, \
    bss_pow_trial_avg_layers, bss_pow_ratio_trial_avg_layers, \
    bss_pow_best_layers, bss_pow_ratio_best_layers = [], [], [], [], [], []
    num_samples_computed = 0
    i_subset = 1

    while num_samples_computed < num_samples:
        run_num_samples = min(num_samples_atonce, num_samples - num_samples_computed)
        indices = [list(range(num_samples_computed, num_samples_computed + run_num_samples))]
        scenario_subset = channel.scenario_select_index(scenario, indices)
        print("Sample set {} of {}:".format(i_subset, num_subsets))

        wrates_trial_avg_layers_subset, wrates_best_layers_subset, \
        bss_pow_trial_avg_layers_subset, bss_pow_ratio_trial_avg_layers_subset, \
        bss_pow_best_layers_subset, bss_pow_ratio_best_layers_subset = \
            downlink_wmmse50(scenario_subset, num_iter=num_iter, num_trials=num_trials, use_stable=use_stable)

        wrates_trial_avg_layers.append(wrates_trial_avg_layers_subset)
        wrates_best_layers.append(wrates_best_layers_subset)
        bss_pow_trial_avg_layers.append(bss_pow_trial_avg_layers_subset)
        bss_pow_ratio_trial_avg_layers.append(bss_pow_ratio_trial_avg_layers_subset)
        bss_pow_best_layers.append(bss_pow_best_layers_subset)
        bss_pow_ratio_best_layers.append(bss_pow_ratio_best_layers_subset)

        num_samples_computed += run_num_samples
        i_subset += 1

    wrates_trial_avg_layers = torch.cat(wrates_trial_avg_layers, dim=-1)
    wrates_best_layers = torch.cat(wrates_best_layers, dim=-1)
    bss_pow_trial_avg_layers = torch.cat(bss_pow_trial_avg_layers, dim=-2)
    bss_pow_ratio_trial_avg_layers = torch.cat(bss_pow_ratio_trial_avg_layers, dim=-2)
    bss_pow_best_layers = torch.cat(bss_pow_best_layers, dim=-2)
    bss_pow_ratio_best_layers = torch.cat(bss_pow_ratio_best_layers, dim=-2)
    # print(bss_pow_best_layers[-1].mean())
    return wrates_trial_avg_layers, wrates_best_layers, \
           bss_pow_trial_avg_layers, bss_pow_ratio_trial_avg_layers, \
           bss_pow_best_layers, bss_pow_ratio_best_layers


def large_scenario_set_initialized_downlink_wmmse(scenario, init="mrc", num_iter=100, use_stable=True):
    """
    Automatically splits up a scenario batch and passes it to the WMMSE algorithm due to memory constraints.
    """
    num_samples = scenario["channels"][0][0].size(0)  # scenario["batch_size"]
    bss_dim = scenario["bss_dim"][0].item()
    num_trials_atonce = 100000 / (num_samples * num_iter * ((bss_dim**2) / 32))

    num_samples_atonce = math.floor(num_samples * num_trials_atonce)
    num_subsets = math.ceil(num_samples / num_samples_atonce)
    wrates_layers, bss_pow_layers, bss_pow_ratio_layers, = [], [], []

    num_samples_computed = 0
    i_subset = 1
    while num_samples_computed < num_samples:
        run_num_samples = min(num_samples_atonce, num_samples - num_samples_computed)
        indices = [list(range(num_samples_computed, num_samples_computed + run_num_samples))]
        scenario_subset = channel.scenario_select_index(scenario, indices)
        print("Sample set {} of {}:".format(i_subset, num_subsets))
        csi = channel.prepare_csi(scenario_subset)
        if init == "mrc":
            if use_stable:
                dlbeamformer_bf_subset, _, _, bss_pow_layers_subset = \
                    downlink_wmmse_stable(csi, init_dl_beamformer=downlink_mrc(scenario_subset), num_iter=num_iter)
            else:
                dlbeamformer_bf_subset, _, _, bss_pow_layers_subset = \
                    downlink_wmmse(csi, init_dl_beamformer=downlink_mrc(scenario_subset), num_iter=num_iter)
        elif init == "zf":
            if use_stable:
                dlbeamformer_bf_subset, _, _, bss_pow_layers_subset = \
                    downlink_wmmse_stable(csi, init_dl_beamformer=downlink_zf_bf(scenario_subset), num_iter=num_iter)
            else:
                dlbeamformer_bf_subset, _, _, bss_pow_layers_subset = \
                    downlink_wmmse(csi, init_dl_beamformer=downlink_zf_bf(scenario_subset), num_iter=num_iter)
        else:
            raise ValueError
        _, wrates_layers_subset = channel.downlink_sum_rate(scenario_subset, dlbeamformer_bf_subset)
        bss_pow_ratio_layers_subset = bss_pow_layers_subset / scenario_subset["bss_pow"].unsqueeze(0)

        wrates_layers.append(wrates_layers_subset)
        bss_pow_layers.append(bss_pow_layers_subset)
        bss_pow_ratio_layers.append(bss_pow_ratio_layers_subset)

        num_samples_computed += run_num_samples
        i_subset += 1

    wrates_layers = torch.cat(wrates_layers, dim=-1)
    bss_pow_layers = torch.cat(bss_pow_layers, dim=-2)
    bss_pow_ratio_layers = torch.cat(bss_pow_ratio_layers, dim=-2)

    return wrates_layers, bss_pow_layers, bss_pow_ratio_layers


def downlink_wmmse50_batchavg(scenario, num_iter=25, num_trials=100):
    wrates_sopt, wrates_avg_layers, bss_pow_avg_layers, bss_pow_ratio_avg_layers, _, _ = downlink_wmmse50(scenario, num_iter=num_iter, num_trials=num_trials)
    batch_size = list(wrates_sopt.size())
    batch_ndim = len(batch_size)

    wrates_sopt = torch.mean(wrates_sopt)
    # wrates_avg = torch.mean(wrates_avg)
    for i_dim in range(batch_ndim):
        wrates_avg_layers = torch.mean(wrates_avg_layers, dim=1)
        bss_pow_avg_layers = torch.mean(bss_pow_avg_layers, dim=1)
        bss_pow_ratio_avg_layers = torch.mean(bss_pow_ratio_avg_layers, dim=1)

    return wrates_sopt, wrates_avg_layers, bss_pow_avg_layers, bss_pow_ratio_avg_layers


def downlink_wmmse_stable(scenario, init_dl_beamformer=[], num_iter=100, show_iteration=False):
    """
    WMMSE algorithm that completely avoids the eigendecomposition, but finds mu
    by calculating the power directly and performing bisection search (more reliable for #TxAnt > #RxAnt).
    :param scenario:
    :param init_dl_beamformer: iterable of matrices, initial guess for the downlink beamformer, matrices are in batch-form
    :param num_iter: number of iterations
    :return:
    """
    num_layer = num_iter

    # Internal configs
    lagrangian_max_iter = 50

    if not init_dl_beamformer:
        init_dl_beamformer = downlink_randn_bf(scenario)

    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    assigned_bs = scenario["users_assign"]
    rweights = scenario["rate_weights"]

    bss_dim = scenario["bss_dim"]
    users_dim = scenario["users_dim"]
    bss_maxpow = scenario["bss_pow"]

    batch_size = list(init_dl_beamformer[0].size())[0:-2]
    batch_ndim = init_dl_beamformer[0].ndim - 2  # num dimension before matrix dimensions
    expand_dim = [1] * batch_ndim

    rtype = scenario["bss_pow"].dtype

    # prepare and expand channels
    channels = []
    for i_bs in range(num_bss):
        channels.append([])
        for i_user in range(num_users):
            channels[i_bs].append(scenario["channels"][i_bs][i_user].expand([*batch_size, *list(scenario["channels"][i_bs][i_user].size())[-2:]]))  # extends channels into batch dim

    # layer or iteration dimension will be added as first dimension for all tensors.
    v, u, w = [], [], []
    for i_user in range(num_users):
        v.append([])
        u.append([])
        w.append([])
        v[i_user].append(init_dl_beamformer[i_user])

    for i_layer in range(num_layer):
        # MMSE: U + W for every user
        # U * W * U^H is equal for every BS, it will be calculated concurrently as preparation for calculation of V
        uwu = []
        v_tilde = []  # preps for calc of V
        for i_user in range(num_users):
            # SUM(H * V * V^H * H) + sigma * I
            user_dim = scenario["users_dim"][i_user]
            eye = torch.eye(user_dim, device=device).view(*expand_dim, user_dim, user_dim)
            dl_covariance_mat = scenario["users_noise_pow"][..., i_user].unsqueeze(-1).unsqueeze(-1) * eye

            # SUM efficiently calculated
            for i_bs in range(num_bss):
                bs_assigned_users = scenario["bss_assign"][i_bs]
                v_stack = torch.stack([v[lj][i_layer] for lj in bs_assigned_users], dim=0)
                bs_channel2user = channels[i_bs][i_user]
                partial_covariance_mats = torch.matmul(bs_channel2user, v_stack)
                partial_covariance_mats = util.cmat_square(partial_covariance_mats)
                dl_covariance_mat = dl_covariance_mat + partial_covariance_mats.sum(dim=0)

            # U = INV * H * V
            inv_cov_mat = util.clean_hermitian(torch.linalg.inv(util.clean_hermitian(dl_covariance_mat)))  # second clean did not help
            user_channel = channels[assigned_bs[i_user]][i_user]
            u_temp = torch.matmul(inv_cov_mat, user_channel)  # takes channel mat of correct BS
            u_temp = torch.matmul(u_temp, v[i_user][i_layer])
            u[i_user].append(u_temp)

            # W = INV(I - U^H * H * V)
            w_inv = torch.matmul(u_temp.conj().transpose(-2, -1), user_channel)
            w_inv = util.clean_hermitian(torch.matmul(w_inv, v[i_user][i_layer]))
            w_inv = eye - w_inv
            w_temp = util.clean_hermitian(torch.linalg.inv(w_inv))
            w[i_user].append(w_temp)

            # U * W * U^H prepares for calc of V
            uwu_temp = torch.matmul(u_temp, w_temp)
            uwu_temp = util.clean_hermitian(torch.matmul(uwu_temp, u_temp.conj().transpose(-2, -1)))  # matrix must be hermitian
            uwu.append(uwu_temp)

            # Preperation V_tilde
            v_tilde_temp = util.mmchain(user_channel.conj().transpose(-2, -1), u_temp, w_temp)
            v_tilde_temp = v_tilde_temp * rweights[..., i_user].unsqueeze(-1).unsqueeze(-1)
            v_tilde.append(v_tilde_temp)

        "CALCULATION OF V"
        bss_current_pow = torch.zeros(*batch_size, num_bss, device=device)
        for i_bs in range(num_bss):

            # collecting user mat
            v_tilde_bsmat = []
            for i_user in scenario["bss_assign"][i_bs]:  # every user assigned to i_bs
                v_tilde_bsmat.append(v_tilde[i_user])
            v_tilde_bsmat = torch.cat(v_tilde_bsmat, dim=-1)

            # Calculation of UL Covariance SUM
            partial_ul_cov_mats = []
            for i_user in range(num_users):
                bs_channel2user = channels[i_bs][i_user]
                # alpha * H^H * U * W * U^H * H
                partial_ul_cov_mat_temp = torch.matmul(bs_channel2user.conj().transpose(-2, -1), uwu[i_user])
                partial_ul_cov_mat_temp = torch.matmul(partial_ul_cov_mat_temp, bs_channel2user)

                partial_ul_cov_mat_temp = util.clean_hermitian(partial_ul_cov_mat_temp) * rweights[..., i_user].unsqueeze(-1).unsqueeze(-1)
                partial_ul_cov_mats.append(partial_ul_cov_mat_temp)

            ul_cov_mat_temp = torch.stack(partial_ul_cov_mats, dim=0).sum(dim=0)  # sums all partial mats

            # User pow mat for upper bound
            aux_assigned_user_mat = []
            for i_user in scenario["bss_assign"][i_bs]:  # every user assigned to i_bs
                v_tilde_temp = v_tilde[i_user]
                aux_assigned_user_mat_temp = util.cmat_square(v_tilde_temp)
                aux_assigned_user_mat.append(aux_assigned_user_mat_temp)
            aux_assigned_user_mat = torch.stack(aux_assigned_user_mat, dim=0).sum(dim=0)

            eigenval_temp, eigenvec_temp = torch.linalg.eigh(ul_cov_mat_temp)
            eigenvec_temp = eigenvec_temp.unsqueeze(0).transpose(0, -1)  # .type(torch.complex64)  # the matrix dim now only hold single eigenvecs, while dim 0 now iterates over different eigenvecs, so (M, *batchdim, M, 1)
            nominator_coeff = util.mmchain(eigenvec_temp.conj().transpose(-2, -1), aux_assigned_user_mat.unsqueeze(0), eigenvec_temp).real.squeeze(-1).squeeze(-1)  # dim 0 of size M holds coefficients, so (M, *batch_size, 1)
            nominator_coeff = nominator_coeff / bss_maxpow[..., i_bs].expand(1, *batch_size)

            mu_root_max = torch.sqrt(nominator_coeff.sum(dim=0))
            mu_root = mu_root_max/2
            mu_step_size = mu_root_max/4
            eye = torch.eye(bss_dim[i_bs], device=device).view(*expand_dim, bss_dim[i_bs], bss_dim[i_bs])
            for ii in range(lagrangian_max_iter):
                inv_augmented_ul_cov_mat_temp = util.clean_hermitian(torch.linalg.inv(ul_cov_mat_temp + eye * mu_root.unsqueeze(-1).unsqueeze(-1)))

                v_running_bsmat = torch.matmul(inv_augmented_ul_cov_mat_temp, v_tilde_bsmat)

                direction = torch.sgn(util.bf_mat_pow(v_running_bsmat) - bss_maxpow[..., i_bs])
                mu_root = mu_root + direction * mu_step_size
                # print(mu_root)
                mu_step_size = mu_step_size/2

            # Decompose to user bf:
            col = 0
            for i_user in scenario["bss_assign"][i_bs]:  # every user assigned to i_bs
                num_cols = users_dim[i_user]
                v_temp = v_running_bsmat[..., col:(col+num_cols)]
                v[i_user].append(v_temp)
                col += num_cols
                bss_current_pow[..., i_bs] += util.bf_mat_pow(v_temp)

        # Power clamping in case solving QCQP still fails
        bss_current_pow = torch.maximum(bss_maxpow, bss_current_pow)  # pre clamping, to avoid any division underflows
        # print(bss_current_pow)
        correction_factor = torch.clamp(torch.sqrt(bss_maxpow / bss_current_pow), max=1)
        # print(correction_factor)

        for i_user in range(num_users):
            i_bs = assigned_bs[i_user]
            v[i_user][i_layer+1] = v[i_user][i_layer+1] * correction_factor[..., i_bs].unsqueeze(-1).unsqueeze(-1)

    # Stack matrices and check power
    bss_pow = torch.zeros(num_bss, num_layer+1, *batch_size, device=device)
    for i_user in range(num_users):
        v[i_user] = torch.stack(v[i_user], dim=0)
        u[i_user] = torch.stack(u[i_user], dim=0)
        w[i_user] = torch.stack(w[i_user], dim=0)

        i_bs = assigned_bs[i_user]
        bss_pow[i_bs] += util.bf_mat_pow(v[i_user])
    bss_pow = bss_pow.movedim(0, -1)  # moves bss dimension to last, first dim is over layers

    return v, u, w, bss_pow


def downlink_wmmse_sisoadhoc(scenario, init_dl_beamformer=[], num_iter=100, show_iteration=False):
    """
    WMMSE algorithm optimized for wireless topologies with TX-RX pairs and scalar channels.
    :param scenario:
    :param init_dl_beamformer: iterable, initial guess for the downlink beamformer, matrices are in batch-form
    :param num_iter: number of iterations
    :return:
    """
    def scenario_extraction(scenario):
        num_users = scenario["num_users"]
        num_bss = scenario["num_bss"]
        device = scenario["device"]
        assert (num_users == num_bss)
        user_noise_pow = scenario["users_noise_pow"]
        bss_pow = scenario["bss_pow"]

        channels = scenario["channels"]
        batch_size = list(channels[0][0].size())[:-2]

        channel_mat = torch.zeros(*batch_size, num_users, num_bss, dtype=torch.float64, device=device)
        for i_user in range(num_users):
            for i_bs in range(num_bss):
                channel_mat[..., i_user, i_bs] = torch.abs(channels[i_bs][i_user]).squeeze(-1).squeeze(-1)
        return channel_mat, user_noise_pow, bss_pow

    def vectorize(v_list):
        v_vec = torch.cat(v_list, dim=-2)
        return v_vec

    def devectorize(v_vec):
        v_list = list(torch.split(v_vec, 1, dim=-2))
        return v_list

    def u_step(cmat, rx_noise_pow, v):
        u_tilde = torch.diagonal(cmat, dim1=-2, dim2=-1).unsqueeze(-1) * v
        cov = torch.matmul(cmat.square(), v.square()).sum(dim=-1, keepdim=True) + rx_noise_pow.unsqueeze(-1)
        u = u_tilde / cov
        return u

    def w_step(cmat, v, u):
        error = 1 - u * torch.diagonal(cmat, dim1=-2, dim2=-1).unsqueeze(-1) * v
        w = 1 / error
        return w

    def v_step(cmat, tx_pow, u, w):
        v_tilde = u * torch.diagonal(cmat, dim1=-2, dim2=-1).unsqueeze(-1) * w
        ul_cov = torch.square(u) * w
        ul_cov = torch.matmul(torch.square(cmat).transpose(-2, -1), ul_cov).sum(dim=-1, keepdim=True)

        # SISO Adhoc QCQP
        mu = v_tilde / torch.sqrt(tx_pow).unsqueeze(-1) - ul_cov
        mu = torch.clamp(mu, min=0)

        v = v_tilde / (ul_cov + mu)

        return v

    def do_iteration(cmat, tx_maxpow, rx_noise_pow, v_in):
        u_o = u_step(cmat, rx_noise_pow, v_in)
        w_o = w_step(cmat, v_in, u_o)
        v_o = v_step(cmat, tx_maxpow, u_o, w_o)

        return v_o, u_o, w_o

    device = scenario["device"]

    # Init
    if not init_dl_beamformer:
        v = downlink_randn_bf(scenario)
    else:
        v = init_dl_beamformer

    v_layers = [v]
    u_layers = []
    w_layers = []

    # scenario conversion
    channel_mat, user_noise_pow, bss_pow = scenario_extraction(scenario)
    v = torch.abs(vectorize(v))

    for i_iter in range(num_iter):
        if show_iteration:
            print("\rIteration {}...".format(i_iter+1), end="", flush=True)

        v, u, w = do_iteration(channel_mat, bss_pow, user_noise_pow, v)
        v_layers.append(devectorize(v + 1j * 0))
        u_layers.append(devectorize(u + 1j * 0))
        w_layers.append(devectorize(w + 1j * 0))

    # Restructure to list of users
    v_out = []
    u_out = []
    w_out = []
    for v_user in list(map(list, zip(*v_layers))):  # transposes layer and user dim
        v_out.append(torch.stack(v_user, dim=0))  # stacks layers
    for u_user in list(map(list, zip(*u_layers))):  # transposes layer and user dim
        u_out.append(torch.stack(u_user, dim=0))
    for w_user in list(map(list, zip(*w_layers))):  # transposes layer and user dim
        w_out.append(torch.stack(w_user, dim=0))

    # power
    num_users = scenario["num_users"]
    assigned_bs = scenario["users_assign"]
    layer_batch_size = list(v_out[0].size())[0:-2]
    with torch.no_grad():
        bss_pow = torch.zeros(*layer_batch_size, scenario["bss_pow"].size(-1), device=device)
        # print(bss_pow.size())
        for i_user in range(num_users):
            i_bs = assigned_bs[i_user]
            # print(util.bf_mat_pow(v_out[i_user]).size())
            bss_pow[..., i_bs] = bss_pow[..., i_bs] + util.bf_mat_pow(v_out[i_user])

    return v_out, u_out, w_out, bss_pow


def scenario_is_siso_adhoc(scenario):
    """
    Returns true if the passed scenario batch consists of scenarios with pairwise SISO links.
    """
    bss_dim = scenario["bss_dim"]
    users_dim = scenario["users_dim"]

    return torch.all(bss_dim == 1) and torch.all(users_dim == 1) and len(bss_dim) == len(users_dim)
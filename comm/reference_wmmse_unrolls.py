"""
Extended and reconfigurable implementations of IAIDNN, UWMMSE, Unfolded PGD network architectures in PyTorch. See below
for works on which the code is based.
UWMMSE: Chowdhury et al. - 2020 - Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation
Unfolded PGD: Pellaco et al. - 2020 - Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems
IAIDNN: Hu et al. - 2021 - Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import comm.network as network
import comm.algorithm as algo
import comm.mathutil as util


class UWMMSE(nn.Module):
    def __init__(self,
                 num_layers=5,
                 num_features=4,
                 num_gcn_layers=2,
                 device=torch.device("cpu")):
        super(UWMMSE, self).__init__()

        # register parameter
        self.device = device
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_gcn_layers = num_gcn_layers

        layers = []
        for _ in range(num_layers):
            layers.append(UWMMSELayer(self.num_features,
                                      self.num_gcn_layers,
                                      device=self.device))
            self.layers = nn.ModuleList(layers)

    def forward(self, scenario, layers="all", debug=False):
        if layers == "all":
            num_forward_layers = self.num_layers
        elif isinstance(layers, int):
            assert (self.num_layers >= layers)
            num_forward_layers = layers
        else:
            raise ValueError

        # Modified WMMSE
        v = algo.downlink_mrc(scenario)

        v_layers = [v]
        u_layers = []
        w_layers = []

        # scenario conversion
        channel_mat, user_noise_pow, bss_pow = self.scenario_extraction(scenario)
        v = torch.abs(self.vectorize(v))
        for i_layer in range(num_forward_layers):
            if debug:
                print("Layer", i_layer + 1)
            v, u, w = self.layers[i_layer](channel_mat, user_noise_pow, bss_pow, v)
            v_layers.append(self.devectorize(v + 1j * 0))
            u_layers.append(self.devectorize(u + 1j * 0))
            w_layers.append(self.devectorize(w + 1j * 0))

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
        with torch.no_grad():
            bss_pow = torch.zeros(num_forward_layers + 1, *list(scenario["bss_pow"].size()), device=self.device)
            # print(bss_pow.size())
            for i_user in range(num_users):
                i_bs = assigned_bs[i_user]
                # print(util.bf_mat_pow(v_out[i_user]).size())
                bss_pow[..., i_bs] = bss_pow[..., i_bs] + util.bf_mat_pow(v_out[i_user])

        return v_out, u_out, w_out, bss_pow

    @staticmethod
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

    @staticmethod
    def vectorize(v_list):
        v_vec = torch.cat(v_list, dim=-2)
        return v_vec

    @staticmethod
    def devectorize(v_vec):
        v_list = list(torch.split(v_vec, 1, dim=-2))
        return v_list


class UWMMSELayer(nn.Module):
    def __init__(self,
                 num_features,
                 num_gcn_layers,
                 device=torch.device("cpu")):
        super(UWMMSELayer, self).__init__()
        self.num_features = num_features
        self.num_gcn_layers = num_gcn_layers
        self.device = device

        self.gcn_features = [1] + [self.num_features] * (self.num_gcn_layers - 1) + [1]
        self.gcn_act = ["relu"] * (self.num_gcn_layers - 1) + ["sigmoid"]
        self.gcn_a = network.GCN(self.gcn_features, self.num_gcn_layers, 2, self.gcn_act, weightinit="glorot", biasinit=0.1)
        self.gcn_b = network.GCN(self.gcn_features, self.num_gcn_layers, 2, self.gcn_act, weightinit="glorot", biasinit=0.1)

    def forward(self, channel_mat, user_noise_pow, bss_pow, v_in):
        eps = 1e-12

        def u_step(channel_mat, user_noise_pow, v):
            u_tilde = torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * v
            cov = torch.matmul(channel_mat.square(), v.square()).sum(dim=-1, keepdim=True) + user_noise_pow.unsqueeze(-1)
            u = u_tilde / cov
            return u

        def w_step(channel_mat, v, u):
            channel_mat_diag = torch.diag_embed(torch.diagonal(channel_mat, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            gso_ops = torch.stack((channel_mat_diag, channel_mat), dim=-3)
            ones = torch.ones_like(v)

            a = self.gcn_a(ones, gso_ops)
            b = self.gcn_b(ones, gso_ops)

            # print(a)
            error = 1 - u * torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * v

            # print(error)
            w = a / error + b

            return w

        def v_step(channel_mat, bss_pow, u, w):
            v_tilde = u * torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * w
            ul_cov = torch.square(u) * w
            ul_cov = torch.matmul(torch.square(channel_mat).transpose(-2, -1), ul_cov).sum(dim=-1, keepdim=True)
            v = v_tilde / (ul_cov + eps)
            v = torch.clamp(v, min=0)
            v_max = torch.sqrt(bss_pow).unsqueeze(-1)  # same size as v
            overshoot = v - v_max > 0
            v[overshoot] = v_max[overshoot]
            return v

        u_out = u_step(channel_mat, user_noise_pow, v_in)
        w_out = w_step(channel_mat, v_in, u_out)
        v_out = v_step(channel_mat, bss_pow, u_out, w_out)

        return v_out, u_out, w_out


class IAIDNN(nn.Module):
    def __init__(self,
                 bss_dim=16,
                 users_dim=2,
                 num_streams=2,
                 num_bss=1,
                 num_users=8,
                 num_layers=5,
                 improved=False,
                 users_shared_param=False,
                 permutation_equivariant=False,
                 unfold_all_layers=False,
                 device=torch.device("cpu")):
        super(IAIDNN, self).__init__()

        # register parameter
        self.device = device
        self.num_layers = num_layers
        self.improved = improved
        self.bss_dim = bss_dim
        self.users_dim = users_dim
        self.num_streams = num_streams
        self.num_users = num_users
        self.num_bss = num_bss
        self.users_shared_param = users_shared_param
        self.permutation_equivariant = permutation_equivariant
        self.unfold_all_layers = unfold_all_layers

        layers = []
        for i_layer in range(num_layers):
            if i_layer != num_layers - 1 or self.unfold_all_layers:
                v_step_type = "unfolded"
            else:
                v_step_type = "singlebs"
            layers.append(IAIDNNLayer(
                bss_dim=self.bss_dim,
                users_dim=self.users_dim,
                num_streams=self.num_streams,
                num_bss=self.num_bss,
                num_users=self.num_users,
                improved=self.improved,
                v_step_type=v_step_type,  # unfolded, singlebs
                users_shared_param=self.users_shared_param,
                permutation_equivariant=permutation_equivariant,
                device=self.device))
        self.layers = nn.ModuleList(layers)

    def forward(self, scenario, layers="all", debug=False):
        if layers == "all":
            num_forward_layers = self.num_layers
        elif isinstance(layers, int):
            assert (self.num_layers >= layers)
            num_forward_layers = layers
        else:
            raise ValueError

        # modified scenario:
        mod_scenario = {}
        for key, val in scenario.items():
            if not key == "channels":
                mod_scenario[key] = val
            else:
                mod_scenario[key] = self.modchannels(val)

        v = algo.downlink_zf_bf(mod_scenario, mode="iaidnn")

        """##################"""
        # self.debug_forward(scenario, v)
        """##################"""

        v_layers = [v]
        u_layers = []
        w_layers = []
        bss_pow_layers = []
        for i_layer in range(num_forward_layers):
            if debug:
                print("Layer", i_layer + 1)
            v, u, w, bss_pow = self.layers[i_layer](mod_scenario, v)
            v_layers.append(v)
            u_layers.append(u)
            w_layers.append(w)
            bss_pow_layers.append(bss_pow)

        # power correction
        bss_maxpow = scenario["bss_pow"]
        num_users = scenario["num_users"]
        assigned_bs = scenario["users_assign"]

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

        # power check
        with torch.no_grad():
            bss_pow = torch.zeros(num_forward_layers + 1, *list(scenario["bss_pow"].size()), device=self.device)
            # print(bss_pow.size())
            for i_user in range(num_users):
                i_bs = assigned_bs[i_user]
                # print(util.bf_mat_pow(v_out[i_user]).size())
                """if torch.any(torch.isnan(util.bf_mat_pow(v_out[i_user]))):
                    print(v_out[i_user][torch.isnan(util.bf_mat_pow(v_out[i_user]))])
                    exit()"""
                bss_pow[..., i_bs] = bss_pow[..., i_bs] + util.bf_mat_pow(v_out[i_user])

        return v_out, u_out, w_out, bss_pow

    def modchannels(self, channels):
        modified_channels = []
        device = channels[0][0].device
        dtype = torch.complex128
        batch_size = list(channels[0][0].size())[:-2]
        for i_bs in range(len(channels)):
            modified_channels.append([])
            for i_user in range(len(channels[i_bs])):
                H = channels[i_bs][i_user]
                user_dim = H.size()[-2]
                bs_dim = H.size()[-1]
                assert (user_dim <= self.users_dim and bs_dim <= self.bss_dim)
                modH = torch.zeros(*batch_size, self.users_dim, self.bss_dim, dtype=dtype, device=device)
                modH[..., :user_dim, :bs_dim] = H
                modified_channels[i_bs].append(modH)

        # print(channels[0][0][0])
        # print(modified_channels[0][0][0])

        return modified_channels

    def load_parameters_from_file(self, path):
        """
        Loads data generated by original reference code.
        :param path:
        :return:
        """
        fr = open(path, 'rb')
        Xv = pickle.load(fr)
        Yv = pickle.load(fr)
        Zv = pickle.load(fr)
        Ov = pickle.load(fr)

        Xw = pickle.load(fr)
        Yw = pickle.load(fr)
        Zw = pickle.load(fr)

        Xu = pickle.load(fr)
        Yu = pickle.load(fr)
        Zu = pickle.load(fr)
        Ou = pickle.load(fr)
        fr.close()
        print("len", len(Xv), len(Xw))
        assert (len(Xu) == (self.num_layers + 1))
        for i_layer in range(1, self.num_layers + 1):  # original code skips parameters at l=0
            self.layers[i_layer - 1].overwrite_parameters(Xu[i_layer], Yu[i_layer], Zu[i_layer], Ou[i_layer],
                                                          Xw[i_layer], Yw[i_layer], Zw[i_layer],
                                                          Xv[i_layer], Yv[i_layer], Zv[i_layer], Ov[i_layer])
        print("Parameters overwritten from file.")


class IAIDNNLayer(nn.Module):
    def __init__(self,
                 bss_dim,
                 users_dim,
                 num_streams,
                 num_bss,
                 num_users,
                 improved=False,
                 v_step_type="unfolded",  # unfolded, normal
                 users_shared_param=False,
                 permutation_equivariant=False,
                 device=torch.device("cpu")):
        super(IAIDNNLayer, self).__init__()

        self.device = device
        self.improved = improved
        self.v_step_type = v_step_type
        self.bss_dim = bss_dim
        self.users_dim = users_dim
        self.num_streams = num_streams
        self.num_users = num_users
        self.num_bss = num_bss
        self.users_shared_param = users_shared_param
        self.permutation_equivariant = permutation_equivariant

        if self.users_shared_param:
            num_param = 1
        else:
            num_param = num_users

        dtype = torch.float64
        ctype = torch.complex128

        # These values are obtained from the original authors codebase for a faithful implementation.
        scale_X = 0.1
        if improved:
            scale_Y = 0.1
        else:
            scale_Y = 0.1
        scale_Z = 0.1
        scale_P = 0.1
        scale_O = 0.1

        if not self.permutation_equivariant:
            self.Xu = nn.Parameter(scale_X * util.randc(num_param, users_dim, users_dim, device=self.device, dtype=dtype))
            self.Yu = nn.Parameter(scale_Y * util.randc(num_param, users_dim, users_dim, device=self.device, dtype=dtype))
            self.Zu = nn.Parameter(scale_Z * util.randc(num_param, users_dim, users_dim, device=self.device, dtype=dtype))
            self.Ou = nn.Parameter(scale_O * util.randc(num_param, users_dim, num_streams, device=self.device, dtype=dtype))

            self.Xw = nn.Parameter(
                scale_X * util.randc(num_param, num_streams, num_streams, device=self.device, dtype=dtype))
            self.Yw = nn.Parameter(
                scale_Y * util.randc(num_param, num_streams, num_streams, device=self.device, dtype=dtype))
            self.Zw = nn.Parameter(
                scale_Z * util.randc(num_param, num_streams, num_streams, device=self.device, dtype=dtype))

            if self.improved:
                self.Pu = nn.Parameter(
                    scale_P * util.randc(num_param, users_dim, users_dim, device=self.device, dtype=dtype))
                self.Pw = nn.Parameter(
                    scale_P * util.randc(num_param, num_streams, num_streams, device=self.device, dtype=dtype))

            if self.v_step_type == "unfolded":
                self.Xv = nn.Parameter(scale_X * util.randc(num_param, bss_dim, bss_dim, device=self.device, dtype=dtype))
                self.Yv = nn.Parameter(scale_Y * util.randc(num_param, bss_dim, bss_dim, device=self.device, dtype=dtype))
                self.Zv = nn.Parameter(scale_Z * util.randc(num_param, bss_dim, bss_dim, device=self.device, dtype=dtype))
                self.Ov = nn.Parameter(
                    scale_O * util.randc(num_param, bss_dim, num_streams, device=self.device, dtype=dtype))
                if self.improved:
                    self.Pv = nn.Parameter(
                        scale_P * util.randc(num_param, bss_dim, bss_dim, device=self.device, dtype=dtype))
            else:
                raise ValueError

        else:
            # no P for improvement included
            self.dXu = nn.Parameter(scale_X * util.randc(num_param, 1, device=self.device, dtype=dtype))
            self.dYu = nn.Parameter(scale_Y * util.randc(num_param, 1, device=self.device, dtype=dtype))
            self.Zu = nn.Parameter(scale_Z * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))
            self.Ou = nn.Parameter(scale_O * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

            self.oXu = nn.Parameter(scale_X * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))
            self.oYu = nn.Parameter(scale_Y * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

            self.Xu = None  # placeholder
            self.Yu = None  # placeholder

            self.dXw = nn.Parameter(
                scale_X * util.randc(num_param, 1, device=self.device, dtype=dtype))
            self.dYw = nn.Parameter(
                scale_Y * util.randc(num_param, 1, device=self.device, dtype=dtype))
            self.Zw = nn.Parameter(
                scale_Z * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

            self.oXw = nn.Parameter(
                scale_X * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))
            self.oYw = nn.Parameter(
                scale_Y * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

            self.Xw = None  # placeholder
            self.Yw = None  # placeholder

            if self.improved:
                self.dPu = nn.Parameter(
                    scale_P * util.randc(num_param, 1, device=self.device, dtype=dtype))
                self.oPu = nn.Parameter(
                    scale_P * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

                self.Pu = None  # placeholder

                self.dPw = nn.Parameter(
                    scale_P * util.randc(num_param, 1, device=self.device, dtype=dtype))
                self.oPw = nn.Parameter(
                    scale_P * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

                self.Pw = None  # placeholder

            if self.v_step_type == "unfolded":
                self.dXv = nn.Parameter(scale_X * util.randc(num_param, 1, device=self.device, dtype=dtype))
                self.dYv = nn.Parameter(scale_Y * util.randc(num_param, 1, device=self.device, dtype=dtype))
                self.Zv = nn.Parameter(scale_Z * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))
                self.Ov = nn.Parameter(
                    scale_O * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

                self.oXv = nn.Parameter(scale_X * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))
                self.oYv = nn.Parameter(scale_Y * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

                self.Xv = None  # placeholder
                self.Yv = None  # placeholder

                if self.improved:
                    self.dPv = nn.Parameter(
                        scale_P * util.randc(num_param, 1, device=self.device, dtype=dtype))
                    self.oPv = nn.Parameter(
                        scale_P * util.randc(num_param, 1, 1, device=self.device, dtype=dtype))

                    self.Pv = None  # placeholder

    def overwrite_parameters(self, Xu, Yu, Zu, Ou, Xw, Yw, Zw, Xv=None, Yv=None, Zv=None, Ov=None):
        ctype = torch.complex128

        def stack_numpy_matrices(matrices):
            array_array = []
            for imat in range(len(matrices)):
                array_array.append(torch.from_numpy(matrices[imat]).to(self.device).type(ctype))
            stacked = torch.stack(array_array, dim=0)
            return stacked

        self.Xu = nn.Parameter(stack_numpy_matrices(Xu))
        self.Yu = nn.Parameter(stack_numpy_matrices(Yu))
        self.Zu = nn.Parameter(stack_numpy_matrices(Zu))
        self.Ou = nn.Parameter(stack_numpy_matrices(Ou))
        self.Xw = nn.Parameter(stack_numpy_matrices(Xw))
        self.Yw = nn.Parameter(stack_numpy_matrices(Yw))
        self.Zw = nn.Parameter(stack_numpy_matrices(Zw))
        if self.v_step_type == "unfolded":
            self.Xv = nn.Parameter(stack_numpy_matrices(Xv))
            self.Yv = nn.Parameter(stack_numpy_matrices(Yv))
            self.Zv = nn.Parameter(stack_numpy_matrices(Zv))
            self.Ov = nn.Parameter(stack_numpy_matrices(Ov))

    def forward(self, scenario, v_in):
        # Internal configs
        eps = 1e-12  # clamp for correction

        device = scenario["device"]
        num_users = scenario["num_users"]
        num_bss = scenario["num_bss"]
        assigned_bs = scenario["users_assign"]
        rweights = scenario["rate_weights"]
        bss_dim = self.bss_dim
        users_dim = self.users_dim
        num_streams = self.num_streams
        bss_maxpow = scenario["bss_pow"]

        batch_size = list(v_in[0].size())[0:-2]
        batch_ndim = len(batch_size)  # num dimension before matrix dimensions
        expand_dim = [1] * batch_ndim
        ctype = scenario["channels"][0][0].dtype
        rtype = scenario["bss_pow"].dtype

        if self.permutation_equivariant:
            # Combination
            self.Xu = torch.diag_embed(self.dXu.expand(num_users, users_dim), dim1=-2, dim2=-1) + self.oXu.expand(num_users, users_dim, users_dim)
            self.Yu = torch.diag_embed(self.dYu.expand(num_users, users_dim), dim1=-2, dim2=-1) + self.oYu.expand(num_users, users_dim, users_dim)

            self.Xw = torch.diag_embed(self.dYw.expand(num_users, num_streams), dim1=-2, dim2=-1) + self.oYw.expand(num_users, num_streams, num_streams)
            self.Yw = torch.diag_embed(self.dYw.expand(num_users, num_streams), dim1=-2, dim2=-1) + self.oYw.expand(num_users, num_streams, num_streams)

            if self.improved:
                self.Pu = torch.diag_embed(self.dPu.expand(num_users, num_streams), dim1=-2, dim2=-1) + self.oPu.expand(num_users, num_streams, num_streams)
                self.Pw = torch.diag_embed(self.dPw.expand(num_users, num_streams), dim1=-2, dim2=-1) + self.oPw.expand(num_users, num_streams, num_streams)

        # Expanding in case of shared param
        Xu = self.Xu.expand(num_users, users_dim, users_dim)
        Yu = self.Yu.expand(num_users, users_dim, users_dim)
        Zu = self.Zu.expand(num_users, users_dim, users_dim)
        Ou = self.Ou.expand(num_users, users_dim, num_streams)

        Xw = self.Xw.expand(num_users, num_streams, num_streams)
        Yw = self.Yw.expand(num_users, num_streams, num_streams)
        Zw = self.Zw.expand(num_users, num_streams, num_streams)

        if self.improved:
            Pu = self.Pu.expand(num_users, users_dim, users_dim)
            Pw = self.Pw.expand(num_users, num_streams, num_streams)

        if self.v_step_type == "unfolded":
            if self.permutation_equivariant:
                self.Xv = torch.diag_embed(self.dYv.expand(num_users, bss_dim), dim1=-2, dim2=-1) + self.oYv.expand(num_users, bss_dim, bss_dim)
                self.Yv = torch.diag_embed(self.dYv.expand(num_users, bss_dim), dim1=-2, dim2=-1) + self.oYv.expand(num_users, bss_dim, bss_dim)

                if self.improved:
                    self.Pv = torch.diag_embed(self.dYv.expand(num_users, bss_dim), dim1=-2, dim2=-1) + self.oPv.expand(num_users, bss_dim, bss_dim)

            Xv = self.Xv.expand(num_users, bss_dim, bss_dim)
            Yv = self.Yv.expand(num_users, bss_dim, bss_dim)
            Zv = self.Zv.expand(num_users, bss_dim, bss_dim)
            Ov = self.Ov.expand(num_users, bss_dim, num_streams)

            if self.improved:
                Pv = self.Pv.expand(num_users, bss_dim, bss_dim)

        elif self.v_step_type == "normal":
            pass
        else:
            raise ValueError

        channels = []
        for i_bs in range(num_bss):
            channels.append([])
            for i_user in range(num_users):
                channels[i_bs].append(scenario["channels"][i_bs][i_user].expand([*batch_size, *list(
                    scenario["channels"][i_bs][i_user].size())[-2:]]))  # extends channels into batch dim

        def uw_step():
            u_out = []
            w_out = []
            uwu = []
            v_tilde = []
            """U Step"""

            # Power in
            bs_pow_in = torch.zeros_like(bss_maxpow)
            for i_user in range(num_users):
                user_bs = assigned_bs[i_user]
                bs_pow_in[..., user_bs] = bs_pow_in[..., user_bs] + util.bf_mat_pow(v_in[i_user])

            for i_user in range(num_users):
                user_bs = assigned_bs[i_user]

                A = torch.eye(users_dim, device=device).view(*expand_dim, users_dim, users_dim) * \
                    scenario["users_noise_pow"][..., i_user].unsqueeze(-1).unsqueeze(-1) * \
                    bs_pow_in[..., user_bs].unsqueeze(-1).unsqueeze(-1) / \
                    bss_maxpow[..., user_bs].unsqueeze(-1).unsqueeze(-1)  # theoretically cancels with previous terms

                for i_bs in range(num_bss):
                    bs_assigned_users = scenario["bss_assign"][i_bs]
                    v_stack = torch.stack([v_in[lj] for lj in bs_assigned_users], dim=0)
                    bs_channel2user = channels[i_bs][i_user]
                    # bs_channel2user = bs_channel2user.expand(1, *batch_size, *list(bs_channel2user.size())[-2:])  # unsqueezes into same dimensions as v_stack
                    partial_covariance_mats = torch.matmul(bs_channel2user, v_stack)
                    partial_covariance_mats = util.cmat_square(partial_covariance_mats)
                    A = A + partial_covariance_mats.sum(dim=0)

                # print(A[0])

                user_channel = channels[user_bs][i_user]
                u_tilde = torch.matmul(user_channel, v_in[i_user])

                Aplus = torch.diag_embed(1 / torch.diagonal(A, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
                Asum = torch.matmul(Aplus, Xu[i_user])
                if self.improved:
                    Ainv = torch.linalg.inv(A)
                    Asum = Asum + util.mmchain(Pu[i_user], Ainv, Yu[i_user])
                else:
                    pass
                    # Asum = Asum + torch.matmul(A, Yu[i_user])
                Asum = Asum + Zu[i_user]
                u_temp = torch.matmul(Asum, u_tilde) + Ou[i_user]
                u_out.append(u_temp)
                # print(u_temp[0])
                """W Step"""
                eye = torch.eye(num_streams, device=device).view(*expand_dim, num_streams, num_streams)
                E = eye - util.mmchain(u_temp.conj().transpose(-2, -1), user_channel, v_in[i_user])

                Eplus = torch.diag_embed(1 / torch.diagonal(E, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
                w_temp = torch.matmul(Eplus, Xw[i_user])
                if self.improved:
                    # print(E)
                    Einv = torch.linalg.inv(E)
                    w_temp = w_temp + util.mmchain(Pw[i_user], Einv, Yw[i_user])
                else:
                    pass
                    # w_temp = w_temp + torch.matmul(E, Yw[i_user])
                w_temp = w_temp + Zw[i_user]

                w_out.append(w_temp)
                # print(w_temp[0])

                """UWU"""
                uwu.append(util.mmchain(u_temp, w_temp, u_temp.conj().transpose(-2, -1)))

                """Prep v_tilde"""
                v_tilde_temp = util.mmchain(user_channel.conj().transpose(-2, -1), u_temp, w_temp) \
                               * rweights[..., i_user].unsqueeze(-1).unsqueeze(-1)
                v_tilde.append(v_tilde_temp)
            # print("w_temp", w_temp[0])
            # print("u_temp", u_temp[0])
            # print("A", A[0])
            # print("E", E[0])
            # print("uwu", uwu[0][0])
            return u_out, w_out, uwu, v_tilde

        def v_step():

            B = []
            Binv = []
            Bplus = []
            for i_bs in range(num_bss):

                partial_ul_cov_mats = []
                uwu_sum = 0

                # Calculation of UL Covariance SUM # and # SUM(Tr(uwu))
                for i_user in range(num_users):
                    bs_channel2user = channels[i_bs][i_user]
                    # bs_channel2user = bs_channel2user.expand(*batch_size, *list(bs_channel2user.size())[-2:])  # expand to batch dimension
                    # alpha * H^H * U * W * U^H * H
                    partial_ul_cov_mat_temp = torch.matmul(bs_channel2user.conj().transpose(-2, -1), uwu[i_user])
                    partial_ul_cov_mat_temp = torch.matmul(partial_ul_cov_mat_temp, bs_channel2user)

                    partial_ul_cov_mat_temp = partial_ul_cov_mat_temp * rweights[
                        ..., i_user].unsqueeze(-1).unsqueeze(-1)
                    partial_ul_cov_mats.append(partial_ul_cov_mat_temp)

                    uwu_sum = uwu_sum + (util.btrace(uwu[i_user]) * rweights[..., i_user] * scenario["users_noise_pow"][
                        ..., i_user])

                ul_cov_mat_temp = torch.stack(partial_ul_cov_mats, dim=0).sum(dim=0)

                diagload = torch.eye(bss_dim, device=device).view(*expand_dim, bss_dim, bss_dim) / \
                           bss_maxpow[..., i_bs].unsqueeze(-1).unsqueeze(-1)

                B_temp = ul_cov_mat_temp + diagload * uwu_sum.unsqueeze(-1).unsqueeze(-1)

                B.append(B_temp)
                Bplus.append(
                    torch.diag_embed(1 / torch.diagonal(B_temp, dim1=-2, dim2=-1), dim1=-2, dim2=-1))
                if self.improved or self.v_step_type == "singlebs":
                    Binv.append(torch.linalg.inv(B_temp))

            # Calculation of V
            v_out = []
            bss_current_pow = torch.zeros(*batch_size, num_bss, device=device)
            for i_user in range(num_users):
                i_bs = assigned_bs[i_user]
                if self.v_step_type == "unfolded":
                    Bsum = torch.matmul(Bplus[i_bs], Xv[i_user])

                    if self.improved:
                        Bsum = Bsum + util.mmchain(Pv[i_user], Binv[i_bs], Yv[i_user])
                    else:
                        pass
                        # Bsum = Bsum + util.mmchain(B[i_bs], Yv[i_user])
                    Bsum = Bsum + Zv[i_user]

                    v_temp = torch.matmul(Bsum, v_tilde[i_user]) + Ov[i_user]

                elif self.v_step_type == "singlebs":
                    v_temp = torch.matmul(Binv[i_bs], v_tilde[i_user])
                    # print(v_temp[0])

                else:
                    raise ValueError
                # print("Bsum", Bsum[0])
                #  exit()
                # print("vtilde", v_tilde[0])
                # print(v_temp[0])
                v_out.append(v_temp)

                bss_current_pow[..., i_bs] += util.bf_mat_pow(v_temp)
            # print(bss_current_pow[0])

            return v_out, bss_current_pow

        u_out, w_out, uwu, v_tilde = uw_step()
        v_out, bss_current_pow = v_step()

        return v_out, u_out, w_out, bss_current_pow


class UnfoldedPGD(nn.Module):
    def __init__(self,
                 num_layers=5,
                 num_pgd_steps=4,
                 device=torch.device("cpu")):
        super(UnfoldedPGD, self).__init__()

        # register parameter
        self.device = device
        self.num_layers = num_layers
        self.num_pgd_steps = num_pgd_steps

        layers = []
        for _ in range(num_layers):
            layers.append(UnfoldedPGDLayer(self.num_pgd_steps,
                                           device=self.device))
            self.layers = nn.ModuleList(layers)

    def forward(self, scenario, layers="all", debug=False):
        if layers == "all":
            num_forward_layers = self.num_layers
        elif isinstance(layers, int):
            assert (self.num_layers >= layers)
            num_forward_layers = layers
        else:
            raise ValueError

        # Modified WMMSE
        v = algo.downlink_mrc(scenario)

        v_layers = [v]
        u_layers = []
        w_layers = []

        for i_layer in range(num_forward_layers):
            if debug:
                print("Layer", i_layer + 1)
            v, u, w = self.layers[i_layer](scenario, v)
            v_layers.append(v)
            u_layers.append(u)
            w_layers.append(w)

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
        with torch.no_grad():
            bss_pow = torch.zeros(num_forward_layers + 1, *list(scenario["bss_pow"].size()), device=self.device)
            for i_user in range(num_users):
                i_bs = assigned_bs[i_user]
                bss_pow[..., i_bs] = bss_pow[..., i_bs] + util.bf_mat_pow(v_out[i_user])

        return v_out, u_out, w_out, bss_pow


class UnfoldedPGDLayer(nn.Module):
    def __init__(self,
                 num_pgd_steps=4,
                 device=torch.device("cpu")):
        super(UnfoldedPGDLayer, self).__init__()
        self.num_pgd_steps = num_pgd_steps
        self.device = device

        dtype = torch.float64
        self.pgd_step_size = nn.Parameter(torch.ones(self.num_pgd_steps, dtype=dtype, device=self.device))

    def forward(self, scenario, v_in):
        # Internal configs
        eps = 1e-12  # clamp for correction

        device = scenario["device"]
        num_users = scenario["num_users"]
        num_bss = scenario["num_bss"]
        assigned_bs = scenario["users_assign"]
        rweights = scenario["rate_weights"]
        users_noise_pow = scenario["users_noise_pow"]
        users_dim = scenario["users_dim"]
        bss_maxpow = scenario["bss_pow"]

        batch_size = list(v_in[0].size())[0:-2]
        batch_ndim = len(batch_size)  # num dimension before matrix dimensions
        expand_dim = [1] * batch_ndim
        ctype = scenario["channels"][0][0].dtype
        rtype = scenario["bss_pow"].dtype

        channels = []
        for i_bs in range(num_bss):
            channels.append([])
            for i_user in range(num_users):
                channels[i_bs].append(scenario["channels"][i_bs][i_user].expand([*batch_size, *list(
                    scenario["channels"][i_bs][i_user].size())[-2:]]))  # extends channels into batch dim

        def uw_step():
            u_out = []
            w_out = []
            uwu = []
            v_tilde = []

            for i_user in range(num_users):
                user_bs = assigned_bs[i_user]

                user_dim = users_dim[i_user]
                eye = torch.eye(user_dim, device=device).view(*expand_dim, user_dim, user_dim)
                dl_covariance_mat = users_noise_pow[..., i_user].unsqueeze(-1).unsqueeze(-1) * eye

                """U Step"""
                for i_bs in range(num_bss):
                    bs_assigned_users = scenario["bss_assign"][i_bs]
                    v_stack = torch.stack([v_in[lj] for lj in bs_assigned_users], dim=0)
                    bs_channel2user = channels[i_bs][i_user]
                    # bs_channel2user = bs_channel2user.expand(1, *batch_size, *list(bs_channel2user.size())[-2:])  # unsqueezes into same dimensions as v_stack
                    partial_covariance_mats = torch.matmul(bs_channel2user, v_stack)
                    partial_covariance_mats = util.cmat_square(partial_covariance_mats)
                    dl_covariance_mat = dl_covariance_mat + partial_covariance_mats.sum(dim=0)

                user_channel = channels[user_bs][i_user]
                u_tilde = torch.matmul(user_channel, v_in[i_user])
                u_temp = torch.matmul(util.clean_hermitian(torch.linalg.inv(dl_covariance_mat)), u_tilde)
                u_out.append(u_temp)

                """W Step"""
                eye = torch.eye(user_dim, device=device).view(*expand_dim, user_dim, user_dim)
                error_mat = eye - util.mmchain(u_temp.conj().transpose(-2, -1), user_channel, v_in[i_user])
                w_temp = util.clean_hermitian(torch.linalg.inv(error_mat))
                w_out.append(w_temp)

                """UWU^H"""
                uwu.append(util.mmchain(u_temp, w_temp, u_temp.conj().transpose(-2, -1)))

                """Prep v_tilde"""
                v_tilde_temp = util.mmchain(user_channel.conj().transpose(-2, -1), u_temp, w_temp) \
                               * rweights[..., i_user].unsqueeze(-1).unsqueeze(-1)
                v_tilde.append(v_tilde_temp)

            return u_out, w_out, uwu, v_tilde

        def v_step():

            ul_cov_mats = []
            for i_bs in range(num_bss):

                # A in paper, or R = SUM(H^H UWU^H H)
                partial_ul_cov_mats = []
                # Calculation of UL Covariance SUM # and # SUM(Tr(uwu))
                for i_user in range(num_users):
                    bs_channel2user = channels[i_bs][i_user]
                    # alpha * H^H * U * W * U^H * H
                    partial_ul_cov_mat_temp = torch.matmul(bs_channel2user.conj().transpose(-2, -1), uwu[i_user])
                    partial_ul_cov_mat_temp = torch.matmul(partial_ul_cov_mat_temp, bs_channel2user)
                    partial_ul_cov_mat_temp = util.clean_hermitian(partial_ul_cov_mat_temp)
                    partial_ul_cov_mat_temp = partial_ul_cov_mat_temp * rweights[
                        ..., i_user].unsqueeze(-1).unsqueeze(-1)
                    partial_ul_cov_mats.append(partial_ul_cov_mat_temp)

                ul_cov_mat_temp = torch.stack(partial_ul_cov_mats, dim=0).sum(dim=0)
                ul_cov_mats.append(ul_cov_mat_temp)

            bss_maxpow_root = torch.sqrt(bss_maxpow)
            v_out = v_in.copy()

            for i_pgd_step in range(self.num_pgd_steps):
                bss_current_pow = torch.zeros(*batch_size, num_bss, device=device)
                # Gradient step
                for i_user in range(num_users):
                    i_bs = assigned_bs[i_user]

                    descent_direction = 2 * (torch.matmul(ul_cov_mats[i_bs], v_out[i_user]) - v_tilde[i_user])
                    v_temp = v_out[i_user] - self.pgd_step_size[i_pgd_step] * descent_direction
                    v_out[i_user] = v_temp

                    bss_current_pow[..., i_bs] += util.bf_mat_pow(v_temp)

                projection_factor = bss_maxpow_root / (
                            F.relu(torch.sqrt(bss_current_pow) - bss_maxpow_root) + bss_maxpow_root)
                for i_user in range(num_users):
                    i_bs = assigned_bs[i_user]
                    v_out[i_user] = projection_factor[..., i_bs].unsqueeze(-1).unsqueeze(-1) * v_out[i_user]

            return v_out

        u_out, w_out, uwu, v_tilde = uw_step()
        v_out = v_step()

        return v_out, u_out, w_out

import torch
import torch.nn as nn
import torch.nn.functional as F
import comm.mathutil as util


class GCN(nn.Module):
    def __init__(self,
                 num_features,  # list of length num_layers + 1, (in_feat, .., out_feat)
                 num_layers,
                 num_operators,  # number of shift matrices (for example individual exponents of it etc.)
                 activations,  # list of None, relu, sigmoid
                 num_space="real",
                 recomb=False,  # linear recombination of features of last layer, feat_out will then be 1
                 weightinit="normal",  # "normal", "pos"
                 biasinit=0,
                 device=torch.device("cpu"),
                 ):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_operators = num_operators
        self.activations = activations
        self.num_space = num_space
        self.recomb = recomb
        self.weightinit = weightinit
        self.biasinit = biasinit
        self.device = device
        dtype = torch.float64

        self.layers = []
        for i_layer in range(num_layers):
            self.layers.append(GCNLayer(self.num_features[i_layer],
                                        self.num_features[i_layer + 1],
                                        self.num_operators,
                                        num_space=self.num_space,
                                        activation=activations[i_layer],
                                        weightinit=self.weightinit,
                                        biasinit=self.biasinit,
                                        device=self.device))
        self.layers = nn.ModuleList(self.layers)

        if self.recomb:
            if num_space == "real":
                self.recomb_mat = nn.Parameter(
                    torch.randn(self.num_operators, self.num_features_in, 1, device=self.device, dtype=dtype) /
                    torch.sqrt(torch.tensor(self.num_features_in + 1, device=self.device)))
            elif self.num_space == "complex":
                self.recomb_mat = nn.Parameter(
                    util.randcn(self.num_operators, self.num_features_in, 1, device=self.device, dtype=dtype) /
                    torch.sqrt(torch.tensor(self.num_features_in + 1, device=self.device)))
            else:
                raise ValueError
        else:
            self.recomb_mat = None

    def forward(self, x, gso):
        """
        :param x: (*batch_size, N, feat_in)
        :param gso: (*batch_size, num_operators, N, N), can omit batch_size dimensions
        :return: (*batch_size, N, feat_out)
        """
        x_out = x
        for i_layer in range(self.num_layers):
            x_out = self.layers[i_layer](x_out, gso)

        if self.recomb_mat is not None:
            x_out = torch.matmul(x_out, self.recomb_mat)

        return x_out


class GCNLayer(nn.Module):
    def __init__(self,
                 num_features_in,
                 num_features_out,
                 num_operators,
                 num_space="real",
                 activation="relu",
                 weightinit="normal",  # "pos"
                 biasinit=0,
                 device=torch.device("cpu")):
        super(GCNLayer, self).__init__()

        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.num_operators = num_operators
        self.activation = activation
        self.num_space = num_space
        self.weightinit = weightinit
        self.biasinit = biasinit
        self.device = device
        dtype = torch.float64

        if num_space == "real":
            if self.weightinit == "normal":
                self.filter_taps = nn.Parameter(
                    torch.randn(self.num_operators, self.num_features_in, self.num_features_out, device=self.device,
                                dtype=dtype) /
                    torch.sqrt(torch.tensor(self.num_features_in + self.num_features_out, device=self.device)))
            elif self.weightinit == "pos":
                self.filter_taps = nn.Parameter(
                    torch.rand(self.num_operators, self.num_features_in, self.num_features_out, device=self.device,
                               dtype=dtype) /
                    torch.sqrt(torch.tensor(self.num_features_in + self.num_features_out, device=self.device)))
            elif self.weightinit == "glorot":
                self.filter_taps = nn.Parameter(
                    (2 * torch.rand(self.num_operators, self.num_features_in, self.num_features_out, device=self.device,
                                    dtype=dtype) - 1) \
                    * torch.sqrt(torch.tensor(6, device=self.device) / (self.num_features_in + self.num_features_out)))
            else:
                raise ValueError
            self.bias = nn.Parameter(
                torch.ones(self.num_features_out, device=self.device, dtype=dtype) * self.biasinit)
        elif self.num_space == "complex":
            self.filter_taps = nn.Parameter(
                util.randcn(self.num_operators, self.num_features_in, self.num_features_out, device=self.device,
                            dtype=dtype) /
                torch.sqrt(torch.tensor(self.num_features_in + self.num_features_out, device=self.device)))
            self.bias = nn.Parameter(
                (torch.ones(self.num_features_out, device=self.device, dtype=dtype) +
                 1j * torch.ones(self.num_features_out, device=self.device, dtype=dtype))) * self.biasinit
        else:
            raise ValueError

    def forward(self, x, gso):
        """
        :param x: (*batch_size, N, feat_in)
        :param gso: (*batch_size, num_operators, N, N), can omit batch_size dimensions
        :return:
        """
        N = x.size()[-2]
        batch_size = list(x.size())[:-2]
        x_temp = torch.reshape(x, (*batch_size, 1, N, self.num_features_in))
        x_out = util.mmchain(gso, x_temp, self.filter_taps).sum(dim=-3) + self.bias

        if self.activation is None:
            pass
        elif self.activation == "relu":
            if self.num_space == "real":
                x_out = F.relu(x_out)
            elif self.num_space == "complex":
                x_out = util.complex_relu(x_out)
            else:
                raise ValueError
        elif self.activation == "sigmoid":
            x_out = torch.sigmoid(x_out)
        else:
            raise ValueError

        return x_out

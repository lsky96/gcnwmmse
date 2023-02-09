import torch
import comm.channel as channel
import comm.mathutil as util


def wrate_loss(scenario, dl_beamformers, layers="all"):
    """
    Creates a normalized loss for weighted downlink rates given downlink beamformers applied in a scenario.
    :param scenario:
    :param dl_beamformers:
    :param layers: list of indices of layers (first dim of dl_beamformers), which are to be taken into account, can be "all"
    :return: normalized loss tensor
    """
    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    rweights = scenario["rate_weights"]

    _, wrate = channel.downlink_sum_rate(scenario, dl_beamformers)

    if layers == "all":
        batch_size = list(dl_beamformers[0].size())[:-2]
        loss = - wrate.sum()
    elif layers == "all_excludeinit":
        temp = list(dl_beamformers[0].size())[0:-2]
        layers = list(range(1, temp[0]))
        batch_size = temp[1:]
        loss = 0
        for i_layer in layers:  # work around for gradient bug when indexing from lists
            loss = loss - wrate[i_layer].sum()
        loss = loss / len(layers)
    else:
        batch_size = list(dl_beamformers[0].size())[1:-2]
        loss = 0
        for i_layer in layers:  # work around for gradient bug when indexing from lists
            loss = loss - wrate[i_layer].sum()
        loss = loss / len(layers)

    rweights = rweights.expand([*batch_size, num_users])  # in case if input comes from multiple layers etc.
    loss = loss / rweights.sum()

    return loss


def wrate_loss_samplenorm(scenario, dl_beamformers, layers="all"):
    """
    Creates a normalized loss for weighted downlink rates given downlink beamformers applied in a scenario.
    :param scenario:
    :param dl_beamformers:
    :param layers: list of indices of layers (first dim of dl_beamformers), which are to be taken into account, can be "all"
    :return: normalized loss tensor
    """
    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    rweights = scenario["rate_weights"]

    _, wrate = channel.downlink_sum_rate(scenario, dl_beamformers)

    """ADDED FOR TESTING"""
    # global error_indicator
    # wrate = wrate[error_indicator]
    """"""

    if layers == "all":
        batch_size = list(dl_beamformers[0].size())[:-2]
        loss = - wrate
    elif layers == "all_excludeinit":
        temp = list(dl_beamformers[0].size())[0:-2]
        layers = list(range(1, temp[0]))
        batch_size = temp[1:]
        loss = 0
        for i_layer in layers:  # work around for gradient bug when indexing from lists
            loss = loss - wrate[i_layer]
        loss = loss / len(layers)
    else:
        batch_size = list(dl_beamformers[0].size())[1:-2]
        loss = 0
        for i_layer in layers:  # work around for gradient bug when indexing from lists
            loss = loss - wrate[i_layer]
        loss = loss / len(layers)

    mean_loss = torch.mean(loss.detach())  # in order to see training progress, negative
    """ADDED FOR TESTING"""
    loss = loss / torch.clamp(loss.detach(), max=-1e-6)  # note that loss is negative
    # loss = loss / loss.detach()  # sample normalization, positive
    """"""
    loss = loss.sum() * mean_loss  # see training progress, negative

    rweights = rweights.expand([*batch_size, num_users])  # in case if input comes from multiple layers etc.
    loss = loss / rweights.sum()

    return loss


def iaidnn_loss(scenario, dl_beamformers, layers="all"):
    """
    Creates a normalized loss for weighted downlink rates given downlink beamformers applied in a scenario.
    :param scenario:
    :param dl_beamformers:
    :param layers: list of indices of layers (first dim of dl_beamformers), which are to be taken into account, can be "all"
    :return: normalized loss tensor
    """
    device = scenario["device"]
    num_users = scenario["num_users"]
    num_bss = scenario["num_bss"]
    rweights = scenario["rate_weights"]

    _, wrate = channel.downlink_sum_rate_iadnn(scenario, dl_beamformers)

    if layers == "all":
        batch_size = list(dl_beamformers[0].size())[:-2]
        loss = - wrate.sum()
    elif layers == "all_excludeinit":
        temp = list(dl_beamformers[0].size())[0:-2]
        layers = list(range(1, temp[0]))
        batch_size = temp[1:]
        loss = 0
        for i_layer in layers:  # work around for gradient bug when indexing from lists
            loss = loss - wrate[i_layer].sum()
        loss = loss / len(layers)
    else:
        batch_size = list(dl_beamformers[0].size())[1:-2]
        loss = 0
        for i_layer in layers:  # work around for gradient bug when indexing from lists
            loss = loss - wrate[i_layer].sum()
        loss = loss / len(layers)

    rweights = rweights.expand([*batch_size, num_users])  # in case if input comes from multiple layers etc.
    loss = loss / rweights.sum()

    return loss

import torch
import os


def add_milestone_to_scheduler(scheduler, milestones):
    """
    Adds milestones to scheduler. Ignores milestones which are already part of the current milestones of the scheduler.
    :param scheduler:
    :param milestones:
    :return:
    """
    sd = scheduler.state_dict()
    for ms in milestones:
        if ms not in sd["milestones"]:
            sd["milestones"].update([ms])
    scheduler.load_state_dict(sd)


def dump_tensor_as_csv(tensor, name):
    # Dumps a tensor in inplace
    path = os.path.abspath(name)
    if os.path.isfile(path):
        raise RuntimeError("File already exists. Cannot overwrite file.")

    array = tensor.numpy()

import torch


def translation_ae(pr_pose, gt_pose):
    pr_location = pr_pose[:, :-1]
    gt_location = gt_pose[:, :-1]

    return torch.abs(pr_location - gt_location).sum()


def rotation_ae(pr_pose, gt_pose):
    pr_orientation = pr_pose[:, -1]
    gt_orientation = gt_pose[:, -1]

    return torch.abs(pr_orientation - gt_orientation).sum()

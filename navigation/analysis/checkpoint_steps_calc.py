import torch

#cpt_path = "/private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_rgbd_2021_03_23_22_43_10/ckpt.345.pth"
cpt_path = "/private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_gibson0_pretrained_spl_rew2021_10_13_02_03_31/ckpt.94.pth"
pretrained = True
steps_sum = 0
while pretrained:
    st = torch.load(cpt_path)
    pretrained = st['config']["RL"]["DDPPO"]["pretrained"]
    cpt_path = st['config']["RL"]["DDPPO"]["pretrained_weights"]
    steps_sum += st["extra_state"]["step"]
    print(cpt_path)
    print(f"sum steps: {steps_sum}")

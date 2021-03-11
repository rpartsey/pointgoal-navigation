FROM fairembodied/habitat-challenge:testing_2021_habitat_base_docker

COPY agent.py agent.py
COPY config_files/challenge_pointnav2021_eppg_sensor.remote.rgbd.yaml config_files/challenge_pointnav2021.local.rgbd.yaml
COPY config_files/ddppo/ddppo_pointnav.yaml config_files/ddppo/ddppo_pointnav.yaml
COPY habitat_extensions habitat_extensions
COPY odometry odometry
COPY ddppo.ckpt.pth ddppo.ckpt.pth
COPY vo_config.yaml /vo_config.yaml
COPY vo.ckpt.pth /vo.ckpt.pth
COPY submission.sh submission.sh

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "config_files/challenge_pointnav2021.local.rgbd.yaml"
ENV DDPPO_CONFIG_FILE "config_files/ddppo/ddppo_pointnav.yaml"

RUN /bin/bash -c ". activate habitat; pip install ifcfg torch==1.7.1 torchvision==0.8.2 segmentation-models-pytorch==0.1.3"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --ddppo-config-path $DDPPO_CONFIG_FILE --ddppo-checkpoint-path ddppo.ckpt.pth --input-type rgbd"]

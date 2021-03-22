#!/usr/bin/env bash

DOCKER_NAME="pointnav_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run --runtime=nvidia \
    -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \  # OR  -v $(realpath data):/data
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=config_files/challenge_pointnav2021.local.rgbd.yaml" \
    -e "DDPPO_CONFIG_FILE=config_files/ddppo/ddppo_pointnav.yaml" \
    ${DOCKER_NAME}

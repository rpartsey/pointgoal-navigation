import argparse
import os
from collections import OrderedDict
import numpy as np
import random
import time
import torch

import habitat
from habitat.sims import make_sim
from habitat.tasks.nav.nav import (
    PointGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
)
from agent import (
    PPOAgent,
    PPOAgentV2,
    get_vo_config,
    make_model,
    PointGoalEstimator,
    make_transforms,
)
from habitat.core.utils import try_cv2_import
import habitat_baselines_extensions.common.obs_transformers  # noqa required to register Resize obs transform


cv2 = try_cv2_import()

DEVICE = torch.device("cpu")

# CONFIG_PATH = "/home/locobot/oiayn/config.yaml"
# CKPT_PATH = "/home/locobot/oiayn/best_checkpoint_150e.pt"
# CKPT_PATH = "/home/locobot/oiayn/vo_sim2real_e017.pt"
CONFIG_PATH = "/home/locobot/oiayn/v2_config.yaml"
CKPT_PATH = "/home/locobot/oiayn/v2_best_checkpoint_047e.pt"

LIN_SPEED = 0.25 * 1.5  # in meters per second
ANG_SPEED = np.deg2rad(30) * 1.5  # in radians per second
MAX_COLLISIONS = 40

SUCCESS_DISTANCE = 0.36 # 2 x Agent Radius


def vo_cpu_eval_model(config_path, ckpt_path, device=DEVICE):
    vo_config = get_vo_config(config_path, new_keys_allowed=True)
    vo_model = make_model(vo_config.model).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        new_checkpoint[k.replace("module.", "")] = v
    vo_model.load_state_dict(new_checkpoint)
    vo_model.eval()

    pointgoal_estimator = PointGoalEstimator(
        obs_transforms=make_transforms(vo_config.val.dataset.transforms),
        vo_model=vo_model,
        action_embedding_on=vo_config.model.params.action_embedding_size > 0,
        depth_discretization_on=(
            hasattr(vo_config.val.dataset.transforms, "DiscretizeDepth")
            and vo_config.val.dataset.transforms.DiscretizeDepth.params.n_channels > 0
        ),
        rotation_regularization_on=True,
        vertical_flip_on=True,
        device=device,
    )

    return pointgoal_estimator


def wrap_heading(heading):
    if heading >= np.pi:
        heading -= 2 * np.pi
    elif heading <= -np.pi:
        heading += 2 * np.pi

    return heading


class NavEnv:
    def __init__(self, checkpoint_config, vo=False):

        # Import sensor settings from the config
        sim_cfg = checkpoint_config.TASK_CONFIG.SIMULATOR
        config = habitat.get_config()
        config.defrost()
        # config.PYROBOT.SENSORS = checkpoint_config.SENSORS
        config.PYROBOT.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        # config.PYROBOT.RGB_SENSOR.WIDTH = sim_cfg.RGB_SENSOR.WIDTH
        # config.PYROBOT.RGB_SENSOR.HEIGHT = sim_cfg.RGB_SENSOR.HEIGHT
        config.PYROBOT.RGB_SENSOR.WIDTH = 640
        config.PYROBOT.RGB_SENSOR.HEIGHT = 480
        config.PYROBOT.RGB_SENSOR.ORIENTATION = sim_cfg.RGB_SENSOR.ORIENTATION
        # config.PYROBOT.DEPTH_SENSOR.WIDTH = sim_cfg.DEPTH_SENSOR.WIDTH
        # config.PYROBOT.DEPTH_SENSOR.HEIGHT = sim_cfg.DEPTH_SENSOR.HEIGHT
        config.PYROBOT.DEPTH_SENSOR.WIDTH = 640
        config.PYROBOT.DEPTH_SENSOR.HEIGHT = 480
        config.PYROBOT.DEPTH_SENSOR.ORIENTATION = sim_cfg.DEPTH_SENSOR.ORIENTATION
        config.PYROBOT.DEPTH_SENSOR.MAX_DEPTH = sim_cfg.DEPTH_SENSOR.MAX_DEPTH
        config.PYROBOT.DEPTH_SENSOR.MIN_DEPTH = sim_cfg.DEPTH_SENSOR.MIN_DEPTH
        config.PYROBOT.DEPTH_SENSOR.NORMALIZE_DEPTH = (
            sim_cfg.DEPTH_SENSOR.NORMALIZE_DEPTH
        )
        config.freeze()
        print(config.PYROBOT)

        self.vo = vo
        self.pointgoal_from_initial = None
        self._reality = make_sim(id_sim="PyRobot-v0", config=config.PYROBOT)
        self._goal_location = np.array([0.0, 0.0], dtype=np.float32)
        self._last_time = time.time()

    def _pointgoal(self, agent_state, goal):
        """
        Converts LoCoBot egomotion sensor reading into pointgoal format

        :param agent_state:
        :param goal:
        :return:
        """
        agent_x, agent_y, agent_rotation = agent_state
        agent_coordinates = np.array([agent_x, agent_y])
        rho = np.linalg.norm(agent_coordinates - goal)
        theta = (
            np.arctan2(goal[1] - agent_coordinates[1], goal[0] - agent_coordinates[0])
            - agent_rotation
        )
        theta = theta % (2 * np.pi)
        if theta >= np.pi:
            theta -= 2 * np.pi

        return np.array([rho, theta], dtype=np.float32)

    def reset(self, goal_location):
        self._goal_location = np.array(goal_location)
        observations = self._reality.reset()
        base_state = self._get_base_state()

        point_goal_obs = self._pointgoal(base_state, self._goal_location)

        if self.vo:
            pointgoal_key = PointGoalSensor.cls_uuid
        else:
            pointgoal_key = IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            self.pointgoal_from_initial = point_goal_obs

        observations[pointgoal_key] = point_goal_obs

        return observations

    def get_collision_state(self):
        return self._reality.base.get_collision_state()

    def _get_base_state(self):
        base_state = self._reality.base.get_state("odom")
        base_state = np.array(base_state, dtype=np.float32)
        # print("base_state: {:.3f} {:.3f} {:.3f}".format(*base_state))
        return base_state

    def step(self, action):
        if action == 0:
            return None
        assert action in [1, 2, 3]

        agent_x, agent_y, agent_rotation = self._get_base_state()
        error = None
        if action == 1:  # move forward
            count = 0
            while (error is None or error <= 0.25) and count <= 10:
                self._reality._robot.base.set_vel(LIN_SPEED, 0, 0.2)
                new_agent_x, new_agent_y, _ = self._get_base_state()
                error = np.sqrt(
                    (agent_x - new_agent_x) ** 2 + (agent_y - new_agent_y) ** 2
                )

                # Timeout needed for colliding because no displacement would occur
                count += 1
        else:
            while error is None or wrap_heading(error) <= np.deg2rad(30):
                direction = 1 if action == 2 else -1
                self._reality._robot.base.set_vel(0, direction * ANG_SPEED, 0.2)
                _, _, new_agent_rotation = self._get_base_state()
                error = abs(agent_rotation - new_agent_rotation)

        # Pause to prevent visual motion blur from jerky motions
        self._reality._robot.base.set_vel(0.0, 0.0, 1.0)

        observation = self._reality._sensor_suite.get_observations(
            self._reality.get_robot_observations()
        )

        if self.vo:
            observation[PointGoalSensor.cls_uuid] = self.pointgoal_from_initial
        else:
            observation[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ] = self._pointgoal(self._get_base_state(), self._goal_location)

        return observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, required=True)
    parser.add_argument("-v", "--use-vo", action="store_true")
    parser.add_argument("-g", "--goal", default="6.35,4.5")
    parser.add_argument("-n", "--experiment-name", default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Where logs and images will be stored, if desired
    exp_name = args.experiment_name
    if exp_name is not None:
        os.makedirs(f"data/{exp_name}", exist_ok=True)
        log_file = f"data/{exp_name}/{exp_name}.log"
    else:
        log_file = None

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create agent
    checkpoint = torch.load(args.model_path, map_location="cpu")
    config = checkpoint["config"]
    config.defrost()
    config.RANDOM_SEED = args.seed
    config.INPUT_TYPE = "depth"
    config.MODEL_PATH = args.model_path
    config.RL.POLICY.OBS_TRANSFORMS.RESIZE.SIZE = 256
    config.freeze()
    if args.use_vo:
        pointgoal_estimator = vo_cpu_eval_model(CONFIG_PATH, CKPT_PATH)
        agent = PPOAgentV2(config, pointgoal_estimator=pointgoal_estimator)
    else:
        agent = PPOAgent(config, use_gps=True)
    agent.reset()

    # Create environment
    env = NavEnv(config, vo=args.use_vo)

    # Set up episode
    goal_location = np.array([float(i) for i in args.goal.split(",")], dtype=np.float32)
    print("Starting new episode")
    print("Goal location: {}".format(goal_location))
    observation = env.reset(goal_location)

    # Get initial agent state
    agent_state = env._get_base_state()
    if PointGoalSensor.cls_uuid in observation:
        sensor_uuid = PointGoalSensor.cls_uuid
    else:
        sensor_uuid = IntegratedPointGoalGPSAndCompassSensor.cls_uuid
    rho, theta = observation[sensor_uuid]

    collisions = 0
    done = False
    for step in range(config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS):

        # Save images for debugging / visualization
        if exp_name is not None:
            depth = (
                observation["depth"].astype(np.float32) * 65535
            ).astype(np.uint16).reshape(480, 640)
            cv2.imwrite(
                f"data/{exp_name}/{step:03}_depth.png",
                depth,
            )
            cv2.imwrite(
                f"data/{exp_name}/{step:03}_rgb.png",
                cv2.cvtColor(observation["rgb"], cv2.COLOR_BGR2RGB),
            )

        # Make sure depth has a third dim of 1 for channels
        observation["depth"] = (
            observation["depth"]
            .reshape([*observation["depth"].shape[:2], 1])
            .astype(np.float32)
        )

        # Pass observations to agent to get action, then execute action
        if not env.vo:
            observation.pop("rgb")
        action = agent.act(observation)["action"]
        observation = env.step(action)

        # Record metrics
        action_name = ["STOP ", "FWD  ", "LEFT ", "RIGHT"][action]
        if env.vo:
            pred_rho, pred_theta = agent.pred_rho_theta
            stats = (
                f"A: {action_name}\t"
                f"rho (gt/pred/err): "
                f"{rho:0.3f} / {pred_rho:0.3f} / {rho - pred_rho:0.5f}\t"
                f"theta (gt/pred/err): "
                f"{theta:0.3f} / {pred_theta:0.3f} / {(theta - pred_theta):0.3f}\t"
                f"colls: {collisions}\t"
                f"steps: {step + 1}"
            )
        else:
            stats = (
                f"A: {action_name}\t"
                f"rho: {rho:0.3}\t"
                f"theta: {theta:0.3f}\t"
                f"colls: {collisions}\t"
                f"steps: {step + 1}"
            )
        print(stats)
        if exp_name is not None:
            agent_x, agent_y, agent_rotation = agent_state
            stats += f"\tbase: {agent_x},{agent_y},{agent_rotation}\n"
            with open(log_file, "a+") as f:
                f.write(stats)

        # Get agent state after executing action
        agent_state = env._get_base_state()

        # Get ground truth GPS
        rho, theta = env._pointgoal(agent_state, env._goal_location)

        # Termination conditions
        if observation is None:
            print(f"STOP WAS CALLED. Dist: {rho}")
            if rho < SUCCESS_DISTANCE:
                print("SUCCESS!!!!")
            else:
                print("Failed.")
            done = True
        if env.get_collision_state():
            collisions += 1
            if collisions > MAX_COLLISIONS:
                print("Max collisions reached. Exiting.")
                done = True

        # Record metrics at final state
        if done:
            agent_x, agent_y, agent_rotation = agent_state
            stats = (
                f"Final actual pose: {agent_x},{agent_y},{agent_rotation}\n"
                f"Final actual rho theta to goal: {rho},{theta}\n"
            )
            print(stats)
            if exp_name is not None:
               with open(log_file, "a+") as f:
                    f.write(stats)
            return  # script completely resolves here

    print("Max actions reached. Exiting.")


if __name__ == "__main__":
    main()

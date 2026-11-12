# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# The below code is inspired from TD-MPC2 https://github.com/nicklashansen/tdmpc2
# licensed under the MIT License

import copy
import importlib.metadata
import logging

import gym
import numpy as np
from metaworld import policies
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

logger = logging.getLogger(__name__)


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None, camera_fovy: float | None = None):
        """
        MetaWorld environment wrapper.

        Args:
            env: The MetaWorld environment to wrap.
            cfg: Configuration object.
            camera_fovy: Camera field of view in degrees. Lower values = more zoomed in.
                        If None (default), uses the original camera FOV (60 degrees for corner2).
        """
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env.render_mode = "rgb_array"
        self.env.camera_name = self.camera_name
        self.env.width = cfg.task_specification.img_size
        self.env.height = cfg.task_specification.img_size
        self.env._freeze_rand_vec = self.cfg.task_specification.env.freeze_rand_vec
        self.action_dim = self.env.action_space.shape[0]
        # Set camera FOV if specified (for zoom effect)
        if camera_fovy is not None:
            self.set_camera_fovy(camera_fovy)

        self.init_renderer()

    def init_renderer(self):
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer  # noqa

        self.env.mujoco_renderer = MujocoRenderer(
            self.env.model,
            self.env.data,
            self.env.mujoco_renderer.default_cam_config,
            width=self.env.width,
            height=self.env.height,
            max_geom=self.env.mujoco_renderer.max_geom,
            camera_id=None,
            camera_name=self.env.camera_name,
        )

    def set_camera_fovy(self, fovy: float):
        """
        Set camera field of view to control zoom level.

        Args:
            fovy: Field of view in degrees. Lower values = more zoomed in.
                  The default corner2 camera uses 60 degrees.
                  Typical zoom values: 45 (slight zoom), 30 (moderate zoom), 20 (high zoom).
        """
        import mujoco

        camera_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if camera_id >= 0:
            self.env.model.cam_fovy[camera_id] = fovy
        else:
            raise ValueError(f"Camera '{self.camera_name}' not found in model.")

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = obs.astype(np.float32)
        self._last_state = obs.copy()
        info["proprio"] = obs[:4]
        info["state"] = obs
        return obs, info

    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state - cur_state) < 0.3
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def prepare(self, seed, init_state, env_info):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        return self.reset()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.cfg.planner.get("enforce_action_bounds", False):
            low, high = self.action_space.low, self.action_space.high
            if not np.isfinite(action).all() or np.any(action < low - 1e-6) or np.any(action > high + 1e-6):
                raise ValueError("Simulator received an action outside the benchmark bounds")
            # Remove only floating-point roundoff from normalize/denormalize.
            action = np.clip(action, low, high)
        obs, reward, terminated, truncated, info = self.env.step(action.copy())
        if self.cfg.task_specification.done_at_succ and info["success"]:
            terminated = True  # Checked after every raw action, not just a plan batch.
        obs = obs.astype(np.float32)
        self._last_state = obs.copy()
        info["proprio"] = obs[:4]
        info["state"] = obs
        return obs, reward, terminated, truncated, info

    def get_replay_state(self):
        """Save integration inputs and Python history, not just a 39D observation."""
        import mujoco

        if self.cfg.task_specification.task != "mw-reach-wall":
            raise ValueError("Replay state capture is currently validated for reach-wall only")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.env.model, spec), dtype=np.float64)
        mujoco.mj_getState(self.env.model, self.env.data, state, spec)
        history_names = (
            "_target_pos", "_prev_obs", "_last_stable_obs", "_last_rand_vec",
            "_freeze_rand_vec", "seeded_rand_vec", "curr_path_length", "_did_see_sim_exception",
            "obj_init_pos", "obj_init_angle", "init_tcp", "init_left_pad", "init_right_pad",
            "maxReachDist", "maxPushDist", "maxPlacingDist", "liftThresh", "objHeight", "heightTarget",
        )
        return {
            "schema_version": 1,
            "metaworld_version": importlib.metadata.version("metaworld"),
            "mujoco_version": mujoco.__version__,
            "state_spec": int(spec), "integration_state": state,
            "history": {name: copy.deepcopy(getattr(self.env, name)) for name in history_names if hasattr(self.env, name)},
            "model_arrays": {name: getattr(self.env.model, name).copy() for name in ("body_pos", "body_quat", "site_pos", "site_quat")},
            "np_random_state": copy.deepcopy(self.env.np_random.bit_generator.state),
            "last_state": self._last_state.copy(),
        }

    def set_replay_state(self, snapshot):
        """Restore a state captured on the same benchmark/library versions."""
        import mujoco

        if snapshot["metaworld_version"] != importlib.metadata.version("metaworld") or snapshot["mujoco_version"] != mujoco.__version__:
            raise ValueError("Replay requires the captured Meta-World and MuJoCo versions")
        for name, value in snapshot["model_arrays"].items():
            getattr(self.env.model, name)[:] = value
        for name, value in snapshot["history"].items():
            setattr(self.env, name, copy.deepcopy(value))
        self.env.np_random.bit_generator.state = copy.deepcopy(snapshot["np_random_state"])
        self._last_state = snapshot["last_state"].copy()
        spec = snapshot["state_spec"]
        mujoco.mj_setState(self.env.model, self.env.data, snapshot["integration_state"], spec)
        mujoco.mj_forward(self.env.model, self.env.data)
        # Forward computes derived geometry but may update solver warmstart.
        # Restore all integration inputs once more before the next real step.
        mujoco.mj_setState(self.env.model, self.env.data, snapshot["integration_state"], spec)
        return self._last_state.copy()

    def update_env(self, env_info):
        pass

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        result = self.env.render().copy()[::-1]  # flip vertically
        if result.sum() == 0:
            logger.info("Reinitializing render MetaworldWrapper")
            self.init_renderer()
            result = self.env.render().copy()[::-1]
            if result.sum() == 0:
                raise ValueError("Rendering failed: 0 after reinit renderer.")
        return result  # H W 3

    def _get_obs(self):
        return self.env._get_obs()


def make_env(cfg, env_cls=None):
    """
    Make Meta-World environment.
    """
    env_id = cfg.task_specification.task.split("-", 1)[-1] + "-v3" + "-goal-observable"
    all_envs = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    if not cfg.task_specification.task.startswith("mw-") or env_id not in all_envs:
        raise ValueError("Unknown task:", cfg.task_specification.task)
    if env_cls is not None:
        env = env_cls(seed=cfg.meta.seed)
    else:
        # We ALWAYS take this option
        env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.meta.seed)
        logger.info(f"No env_cls so env initialized with seed {cfg.meta.seed} and {env_id=}")
    env.seeded_rand_vec = False
    env = MetaWorldWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env


def task_name_to_policy_name(task_name: str):
    """
    Task name is of the form mw-lever-pull
    while the policy name is of the form SawyerLeverPullV3Policy
    so we need to convert the task name to the policy name
    """
    special_cases = {
        "mw-peg-insert-side": "SawyerPegInsertionSideV3Policy",
    }
    if task_name in special_cases:
        return special_cases[task_name]
    task_name = task_name.split("-")
    policy_name = "Sawyer"
    policy_name += "".join([word.capitalize() for word in task_name[1:]])
    policy_name += "V3Policy"
    return policy_name


def task_name_to_policy(task_name: str):
    policy_name = task_name_to_policy_name(task_name)
    return getattr(policies, policy_name)

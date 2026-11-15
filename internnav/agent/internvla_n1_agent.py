import copy
import os
import threading
from contextlib import nullcontext

import cv2
import imageio
import numpy as np
import torch
from gym import spaces
from PIL import Image

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model import get_config, get_policy
from internnav.model.utils.misc import set_random_seed
from internnav.model.utils.vln_utils import S1Input, S1Output, S2Input, S2Output


from internnav.utils.latent_planner import LatestLatentPlanner


@Agent.register('internvla_n1')
class InternVLAN1Agent(Agent):
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )

    def __init__(self, config: AgentCfg):
        super().__init__(config)
        set_random_seed(0)
        vln_sensor_config = self.config.model_settings
        self._model_settings = ModelCfg(**vln_sensor_config)
        self.device = torch.device(self._model_settings.device)
        self.mode = getattr(self._model_settings, 'infer_mode', 'sync')
        self.sys2_max_forward_step = getattr(self._model_settings, 'sys2_max_forward_step', 8)

        policy = get_policy(self._model_settings.policy_name)
        policy_config = get_config(self._model_settings.policy_name)
        model_config = {'model': self._model_settings.model_dump()}
        self.policy = policy(config=policy_config(model_cfg=model_config))
        self.policy.eval()

        self.camera_intrinsic = self.get_intrinsic_matrix(
            self._model_settings.width, self._model_settings.height, self._model_settings.hfov
        )

        self.episode_step = 0
        self.episode_idx = 0
        self.look_down = False

        # for async dual sys
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.dual_forward_step = 0
        self.sys1_infer_times = 0

        self.sys1_depth_threshold = 5.0
        self.sys1_forward_step = 4

        self.s1_input = S1Input()
        self.s2_input = S2Input()
        self.s2_output = S2Output()
        self.s1_output = S1Output()

        self._planner = LatestLatentPlanner()
        self._step_lock = threading.Lock()
        self._last_s2_step = -self.sys2_max_forward_step

        # vis debug
        self.vis_debug = vln_sensor_config['vis_debug']
        if self.vis_debug:
            self.debug_path = vln_sensor_config['vis_debug_path']
            os.makedirs(self.debug_path, exist_ok=True)
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_dp.mp4", fps=5)
            self.output_pixel = None

    def reset(self, reset_index=None, *, preserve_memory=False):
        with self._step_lock:
            self._planner.reset()
            self._reset_state(reset_index, preserve_memory=preserve_memory)

    def _reset_state(self, reset_index=None, *, preserve_memory=False):
        self._last_s2_step = -self.sys2_max_forward_step
        if reset_index is not None:
            self.episode_idx += 1
            if self.vis_debug:
                self.fps_writer.close()
                self.fps_writer2.close()
        else:
            self.episode_idx = -1

        self.episode_step = 0
        self.look_down = False
        self.s1_input = S1Input()
        self.s2_input = S2Input()
        self.s2_output = S2Output()
        self.s1_output = S1Output()

        # for async dual sys
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.dual_forward_step = 0
        self.sys1_infer_times = 0

        # Reset s2 agent
        self.policy.reset(preserve_memory=preserve_memory)

        if self.vis_debug:
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_dp.mp4", fps=5)

    def get_intrinsic_matrix(self, width, height, hfov) -> np.ndarray:
        width = width
        height = height
        fov = hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        return intrinsic_matrix

    def _infer_plan(self, rgb, depth, pose, instruction, index):
        stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        if stream is not None:
            stream.wait_stream(torch.cuda.default_stream(self.device))
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            try:
                plan = self.policy.s2_step(rgb, depth, pose, instruction, self.camera_intrinsic)
            finally:
                if stream is not None:
                    stream.synchronize()
        plan.idx = index
        plan.rgb_memory = rgb
        plan.depth_memory = depth
        return plan

    def should_infer_s2(self, mode="partial_async"):
        if mode not in ('sync', 'partial_async'):
            raise ValueError(f'Invalid mode: {mode}')
        gap = 1 if mode == 'sync' else self.sys2_max_forward_step
        return not self._planner.busy and self.episode_step - self._last_s2_step >= gap

    def step(self, obs):
        with self._step_lock:
            return self._step(obs)

    def _step(self, obs):
        mode = self.mode  # 'sync', 'partial_async'

        obs = obs[0]  # do not support batch_env currently?
        rgb = obs['rgb']
        depth = obs['depth']
        instruction = obs['instruction']
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        submitted = False
        if self.should_infer_s2(mode):
            submitted = self._planner.submit(
                self._infer_plan, rgb.copy(), depth.copy(), pose.copy(), instruction, self.episode_step
            )
            if submitted:
                self._last_s2_step = self.episode_step
        if not submitted:
            self.policy.step_no_infer(rgb, depth, pose)

        try:
            generation, plan = self._planner.take()
        except RuntimeError:
            self._last_s2_step = -self.sys2_max_forward_step
            raise
        if plan is None:
            return [{'action': [0], 'ideal_flag': True}]
        self.s2_output = plan
        self.output_pixel = copy.deepcopy(plan.output_pixel)
        self.look_down = False
        if mode != 'sync':
            processed_pixel_rgb = (
                np.array(Image.fromarray(plan.rgb_memory).resize((224, 224))) / 255.0
            )
            processed_pixel_depth = (
                np.array(Image.fromarray(plan.depth_memory[:, :, 0]).resize((224, 224))) * 10.0
            )
            processed_pixel_depth[processed_pixel_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold

            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
            processed_depth = (
                np.array(Image.fromarray(depth[:, :, 0]).resize((224, 224))) * 10.0
            )  # should be 0-10m
            processed_depth[processed_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold

            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )  # [1, 2, 224, 224, 3]
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )  # [1, 2, 224, 224, 1]
            self.s1_output = self.policy.s1_step_latent(rgbs, depths, plan.output_latent)
        else:
            self.s1_output = self.policy.s1_step_latent(rgb, depth * 10000.0, plan.output_latent)

        if not self._planner.is_current(generation):
            return [{'action': [0], 'ideal_flag': True}]
        # Execute a short prefix, then replan from the next observation.
        output = {'action': self.s1_output.idx[:1] or [-1]}
        self.sys1_infer_times += 1
        self.dual_forward_step = self.episode_step - self._last_s2_step

        # Visualization
        if self.vis_debug:
            vis = rgb.copy()
            if 'action' in output:
                vis = cv2.putText(vis, str(output['action'][0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.output_pixel is not None:
                pixel = self.output_pixel
                vis = cv2.putText(
                    vis,
                    f"{pixel[1]}, {pixel[0]} ({self.s2_output.idx})",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.circle(vis, (pixel[1], pixel[0]), 5, (0, 255, 0), -1)
                self.output_pixel = None
            self.fps_writer.append_data(vis)

            if self.s1_output.vis_image is not None:
                Image.fromarray(self.s1_output.vis_image).save(
                    os.path.join("./vis_debug_pix/", f"ttttt_{self.episode_step}.png")
                )
                self.fps_writer2.append_data(self.s1_output.vis_image)

        self.episode_step += 1
        if 'action' in output:
            return [{'action': output['action'], 'ideal_flag': True}]
        elif 'velocity' in output:
            return [{'action': output['velocity'], 'ideal_flag': False}]
        else:
            assert False

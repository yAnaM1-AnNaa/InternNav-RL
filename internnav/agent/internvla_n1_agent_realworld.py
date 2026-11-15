import copy
import itertools
import os
import re
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions
from internnav.utils.latent_planner import LatestLatentPlanner

DEFAULT_IMAGE_TOKEN = "<image>"


class InternVLAN1AsyncAgent:
    def __init__(self, args):
        self._planner = LatestLatentPlanner()
        self._step_lock = threading.Lock()
        self._history_lock = threading.RLock()
        self._observation_idx = 0
        self.device = torch.device(args.device)
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"args.model_path: {args.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = 'left'

        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = float('-inf')
        self.memory_latents = None

        # output
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

    def reset(self, *, preserve_memory=False):
        with self._step_lock:
            self._planner.reset()
            self._reset_state(preserve_memory=preserve_memory)

    def _reset_state(self, *, preserve_memory=False):
        self._observation_idx = 0
        self.last_s2_idx = float('-inf')
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        if not preserve_memory:
            self.memory_latents = None

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        with self._history_lock:
            self._record_observation(rgb)

    def _record_observation(self, rgb):
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        self.episode_idx += 1

    def trajectory_tovw(self, trajectory, kp=1.0):
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def _infer_plan(self, rgb, depth, pose, instruction, intrinsic, look_down):
        # Separate CUDA streams allow the independent S1/S2 paths to overlap.
        stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        if stream is not None:
            stream.wait_stream(torch.cuda.default_stream(self.device))
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            try:
                action, latent, pixel = self.step_s2(rgb, depth, pose, instruction, intrinsic, look_down)
            finally:
                if stream is not None:
                    stream.synchronize()
        return S2Output(
            output_action=action,
            output_latent=latent,
            output_pixel=pixel,
            rgb_memory=rgb,
            depth_memory=depth,
        )

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        # HTTP callers and instruction resets must not overlap S1 invocations.
        with self._step_lock:
            return self._step(rgb, depth, pose, instruction, intrinsic, look_down)

    def _step(self, rgb, depth, pose, instruction, intrinsic, look_down):
        self._observation_idx += 1
        due = self._observation_idx - self.last_s2_idx > self.PLAN_STEP_GAP
        submitted = False
        if due or look_down:
            submitted = self._planner.submit(
                self._infer_plan,
                rgb.copy(),
                depth.copy(),
                copy.deepcopy(pose),
                instruction,
                copy.deepcopy(intrinsic),
                look_down,
            )
            if submitted:
                self.last_s2_idx = self._observation_idx
        if not submitted:
            self.step_no_infer(rgb, depth, pose)

        try:
            generation, plan = self._planner.take()
        except RuntimeError:
            self.last_s2_idx = float('-inf')
            raise
        if plan is None:
            return S2Output(output_action=[0])

        # Keep this complete plan snapshot until this S1 inference has finished.
        processed_pixel_rgb = np.array(Image.fromarray(plan.rgb_memory).resize((224, 224))) / 255
        processed_pixel_depth = np.array(Image.fromarray(plan.depth_memory).resize((224, 224)))
        processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
        processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
        rgbs = (
            torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
            .unsqueeze(0)
            .to(self.device)
        )
        depths = (
            torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(self.device)
        )
        with torch.no_grad():
            trajectories = self.step_s1(plan.output_latent, rgbs, depths)
            trajectory = traj_to_actions(trajectories, use_discrate_action=False)
        # STOP received during S1 takes priority over its now-obsolete trajectory.
        if not self._planner.is_current(generation):
            return S2Output(output_action=[0])
        return S2Output(output_trajectory=trajectory, output_pixel=plan.output_pixel)

    def _prepare_s2_inputs(self, rgb, depth, pose, instruction, intrinsic, look_down):
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        else:
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}_look_down.jpg")
        if not look_down:
            self.conversation_history = []
            self.past_key_values = None

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        return inputs, self.episode_idx

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        with self._history_lock:
            inputs, inference_idx = self._prepare_s2_inputs(rgb, depth, pose, instruction, intrinsic, look_down)
        t0 = time.time()
        with torch.no_grad():
            outputs, self.memory_latents = self.model.generate_with_latents(
                **inputs,
                memory_latents=self.memory_latents,
                max_new_tokens=128,
                do_sample=False,
            )
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        with open(f"{self.save_dir}/llm_output_{inference_idx: 04d}.txt", 'w') as f:
            f.write(self.llm_output)
        self.last_output_ids = copy.deepcopy(output_ids[0])
        print(f"output {inference_idx}  {self.llm_output} cost: {t1 - t0}s")
        if re.fullmatch(r'\s*STOP[.!。！]?\s*', self.llm_output, flags=re.IGNORECASE):
            return [0], None, None
        # Coordinates remain diagnostic output; every non-STOP reply supplies a plan.
        coordinates = re.fullmatch(r'\s*(\d+)\s+(\d+)\s*', self.llm_output)
        pixel_goal = None
        if coordinates:
            pixel_goal = [int(coordinates[2]), int(coordinates[1])]
        return None, self.memory_latents, pixel_goal

    def step_s1(self, latent, rgb, depth):
        all_trajs = self.model.generate_traj(latent, rgb, depth)
        return all_trajs

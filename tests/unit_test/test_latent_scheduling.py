"""CPU regression tests for the real scheduling code, without loading VLM weights.

Load the agent classes without their optional simulator/model imports. Their
production method bodies run unchanged with lightweight inference substitutes.
Run with: python -m unittest discover -s tests/unit_test -p test_latent_scheduling.py
"""
import ast
import copy
import importlib.util
import io
import json
import os
import re
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LatestLatentPlanner = load_file('latent_planner', 'internnav/utils/latent_planner.py').LatestLatentPlanner
Task = load_file('instruction_task', 'internnav/utils/instruction_task.py').InstructionTask


class Output(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(output_action=None, output_latent=None, output_pixel=None,
                         output_trajectory=None, rgb_memory=None, depth_memory=None)
        self.__dict__.update(kwargs)


def load_definition(path, name, extra=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    node.decorator_list = []
    namespace = dict(globals(), S2Output=Output, Agent=object,
                     spaces=SimpleNamespace(Box=lambda **kwargs: None), AgentCfg=object)
    namespace.update(extra or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), 'exec'), namespace)
    return namespace[name]


traj_to_actions = load_definition('internnav/model/utils/vln_utils.py', 'traj_to_actions')
RealAgent = load_definition('internnav/agent/internvla_n1_agent_realworld.py', 'InternVLAN1AsyncAgent')
SimAgent = load_definition('internnav/agent/internvla_n1_agent.py', 'InternVLAN1Agent')
create_app = load_definition('scripts/realworld/http_internvla_server.py', 'create_app',
                             {'CAMERA_POSE': np.eye(4), 'CAMERA_INTRINSIC': np.eye(4)})


def plan(number, stop=False):
    return Output(output_latent=None if stop else torch.full((1, 4, 8), float(number)),
                  output_action=[0] if stop else None,
                  rgb_memory=np.full((4, 4, 3), number, dtype=np.uint8),
                  depth_memory=np.full((4, 4), number, dtype=np.float32))


def make_agent():
    agent = RealAgent.__new__(RealAgent)
    agent.device = torch.device('cpu')
    agent._planner = LatestLatentPlanner()
    agent._step_lock = threading.Lock()
    agent._history_lock = threading.RLock()
    agent._observation_idx = 0
    agent.last_s2_idx = 0
    agent.PLAN_STEP_GAP = 1000
    agent.step_no_infer = Mock()
    agent.step_s1 = Mock(return_value=torch.zeros(1, 4, 3))
    return agent


def inputs(value=10):
    return (np.full((4, 4, 3), value, dtype=np.uint8), np.ones((4, 4), dtype=np.float32),
            np.eye(4), 'turn left then find the chair', np.eye(4))


class PlannerTests(unittest.TestCase):
    def test_latest_replaces_pending_and_keeps_active_snapshot(self):
        planner = LatestLatentPlanner()
        first, second, third = plan(1), plan(2), plan(3)
        planner.publish(0, first)
        generation, active = planner.take()
        planner.publish(generation, second)
        planner.publish(generation, third)
        self.assertIs(active, first)
        self.assertIs(planner.take()[1], third)
        self.assertIs(planner.take()[1], third)

    def test_initial_wait_and_reuse_while_worker_busy(self):
        planner = LatestLatentPlanner()
        started, release = threading.Event(), threading.Event()

        def infer():
            started.set()
            self.assertTrue(release.wait(3))
            return plan(2)

        planner.publish(0, plan(1))
        planner.take()
        self.assertTrue(planner.submit(infer))
        self.assertTrue(started.wait(3))
        try:
            self.assertFalse(planner.submit(infer))
            self.assertEqual(planner.take()[1].output_latent[0, 0, 0], 1)
        finally:
            release.set()
            planner._worker.join(3)
        self.assertEqual(planner.take()[1].output_latent[0, 0, 0], 2)

    def test_stop_overrides_pending_and_latches_until_reset(self):
        planner = LatestLatentPlanner()
        planner.publish(0, plan(1))
        generation, _ = planner.take()
        planner.publish(0, plan(2))
        planner.publish(0, plan(0, stop=True))
        planner.publish(0, plan(3))
        self.assertIsNone(planner.take()[1])
        self.assertFalse(planner.is_current(generation))
        self.assertFalse(planner.submit(lambda: plan(4)))
        planner.reset()
        planner.publish(0, plan(5))  # late result from the previous instruction
        planner.publish(1, plan(6))
        self.assertEqual(planner.take()[1].output_latent[0, 0, 0], 6)

    def test_failure_wakes_first_plan_waiter_and_allows_retry(self):
        planner = LatestLatentPlanner()
        planner.submit(Mock(side_effect=ValueError('bad image')))
        with self.assertRaisesRegex(RuntimeError, 'System 2 inference failed'):
            planner.take()
        planner._worker.join(3)
        self.assertTrue(planner.submit(lambda: plan(1)))
        self.assertEqual(planner.take()[1].output_latent[0, 0, 0], 1)
        planner.reset()

    def test_reset_discards_inflight_result(self):
        planner = LatestLatentPlanner()
        started, release = threading.Event(), threading.Event()

        def infer():
            started.set()
            self.assertTrue(release.wait(3))
            return plan(9)

        planner.submit(infer)
        self.assertTrue(started.wait(3))
        with ThreadPoolExecutor(1) as executor:
            resetting = executor.submit(planner.reset)
            try:
                # The generation changes before reset waits for the worker.
                deadline = time.monotonic() + 3
                while planner.is_current(0) and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertFalse(planner.is_current(0))
            finally:
                release.set()
            resetting.result(3)
        planner.publish(1, plan(2))
        self.assertEqual(planner.take()[1].output_latent[0, 0, 0], 2)


class AgentTests(unittest.TestCase):
    def test_s1_keeps_plan_until_finished_then_uses_newest_with_matching_frame(self):
        agent = make_agent()
        agent._planner.publish(0, plan(1))
        started, release = threading.Event(), threading.Event()
        seen = []

        def infer(latent, rgb, depth):
            seen.append((float(latent[0, 0, 0]), float(rgb[0, 0, 0, 0, 0])))
            if len(seen) == 1:
                started.set()
                self.assertTrue(release.wait(3))
            return torch.zeros(1, 4, 3)

        agent.step_s1 = infer
        with ThreadPoolExecutor(1) as executor:
            first = executor.submit(agent.step, *inputs())
            self.assertTrue(started.wait(3))
            try:
                agent._planner.publish(0, plan(2))
                agent._planner.publish(0, plan(3))
            finally:
                release.set()
            self.assertIsNotNone(first.result(3).output_trajectory)
        agent.step(*inputs(20))
        agent.step(*inputs(30))
        self.assertEqual([x[0] for x in seen], [1, 3, 3])
        self.assertAlmostEqual(seen[1][1], 3 / 255)

    def test_stop_received_during_s1_discards_trajectory(self):
        agent = make_agent()
        agent._planner.publish(0, plan(1))

        def infer(*args):
            agent._planner.publish(0, plan(0, stop=True))
            return torch.zeros(1, 4, 3)

        agent.step_s1 = infer
        result = agent.step(*inputs())
        self.assertEqual(result.output_action, [0])
        self.assertIsNone(result.output_trajectory)

    def test_initial_plan_is_generated_without_pixel_goal(self):
        agent = make_agent()
        agent.last_s2_idx = -100
        agent.PLAN_STEP_GAP = 0
        agent.step_s2 = Mock(return_value=(None, plan(7).output_latent, None))
        result = agent.step(*inputs())
        self.assertIsNotNone(result.output_trajectory)
        self.assertIsNone(result.output_action)
        self.assertEqual(agent.step_s1.call_args.args[0][0, 0, 0], 7)
        agent._planner.reset()

    def test_real_s2_routing_keeps_recurrent_memory_for_all_non_stop_outputs(self):
        agent = make_agent()
        agent.memory_latents = None
        class Batch(dict):
            __getattr__ = dict.__getitem__

        prepared = Batch(input_ids=torch.tensor([[1, 2]]))
        agent._prepare_s2_inputs = Mock(return_value=(prepared, 1))
        decoder = Mock()
        agent.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=decoder))
        generation = SimpleNamespace(sequences=torch.tensor([[1, 2, 3]]))
        generator = Mock()
        agent.model = SimpleNamespace(generate_with_latents=generator)
        with tempfile.TemporaryDirectory() as directory:
            agent.save_dir = directory
            previous = None
            for text in ['←←', '↓', 'nothing visible', '123', '12 34', 'do not STOP here', 'STOP']:
                current = torch.randn(1, 4, 8)
                decoder.return_value = text
                generator.return_value = (generation, current)
                action, latent, pixel = agent.step_s2(*inputs())
                self.assertIs(generator.call_args.kwargs['memory_latents'], previous)
                if text == 'STOP':
                    self.assertEqual(action, [0])
                    self.assertIsNone(latent)
                else:
                    self.assertIsNone(action)
                    self.assertIs(latent, current)
                self.assertEqual(agent.llm_output, text)
                if text == '12 34':
                    self.assertEqual(pixel, [34, 12])
                previous = current

    def test_simulation_uses_s1_on_every_step_and_stop_preempts(self):
        agent = SimAgent.__new__(SimAgent)
        agent._planner = LatestLatentPlanner()
        agent._step_lock = threading.Lock()
        agent.mode = 'partial_async'
        agent.device = torch.device('cpu')
        agent.episode_step = 0
        agent._last_s2_step = 0
        agent.sys2_max_forward_step = 100
        agent.sys1_depth_threshold = 5
        agent.sys1_infer_times = 0
        agent.vis_debug = False
        agent.policy = SimpleNamespace(step_no_infer=Mock(),
            s1_step_latent=Mock(return_value=SimpleNamespace(idx=[2, 2, 1])))
        observation = {'rgb': inputs()[0], 'depth': inputs()[1][..., None], 'instruction': 'find chair'}
        first = plan(1)
        first.depth_memory = first.depth_memory[..., None]
        agent._planner.publish(0, first)
        self.assertEqual(agent.step([observation])[0]['action'], [2])
        self.assertEqual(agent.step([observation])[0]['action'], [2])
        self.assertEqual(agent.policy.s1_step_latent.call_count, 2)
        agent._planner.publish(0, plan(0, stop=True))
        self.assertEqual(agent.step([observation])[0]['action'], [0])
        self.assertEqual(agent.policy.s1_step_latent.call_count, 2)


class HttpInstructionTests(unittest.TestCase):
    @staticmethod
    def post(client, reset=False):
        image, depth = io.BytesIO(), io.BytesIO()
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image, format='PNG')
        Image.fromarray(np.zeros((4, 4), dtype=np.uint16)).save(depth, format='PNG')
        image.seek(0)
        depth.seek(0)
        return client.post('/eval_dual', data={
            'json': json.dumps({'reset': reset}), 'image': (image, 'rgb.png'), 'depth': (depth, 'depth.png')})

    def test_stop_advances_instruction_preserving_memory_and_finishes_task(self):
        agent = SimpleNamespace(reset=Mock(), step=Mock(side_effect=[
            Output(output_trajectory=np.zeros((3, 2))), Output(output_action=[0]),
            Output(output_trajectory=np.zeros((3, 2))), Output(output_action=[0])]))
        task = Task('task', {'instructions': {'first': 'find chair', 'second': 'find door'}})
        client = create_app(agent, task).test_client()
        self.assertIn('trajectory', self.post(client, True).json)
        self.assertEqual(task.index, 0)
        self.assertEqual(self.post(client).json, {'discrete_action': [0]})
        self.assertEqual(task.index, 1)
        agent.reset.assert_called_with(preserve_memory=True)
        self.assertIn('trajectory', self.post(client).json)
        self.assertIn('find door', agent.step.call_args.args[3])
        self.assertTrue(self.post(client).json['task_completed'])
        self.assertTrue(task.completed)

    def test_final_step_timeout_returns_completion(self):
        agent = SimpleNamespace(reset=Mock(), step=Mock(return_value=Output(output_trajectory=np.zeros((3, 2)))))
        task = Task('task', {'instructions': {'only': 'find chair'}})
        client = create_app(agent, task).test_client()
        self.post(client, True)
        task.step_deadline = time.monotonic() - 1
        self.assertTrue(self.post(client).json['task_completed'])
        self.assertEqual(agent.step.call_count, 1)


if __name__ == '__main__':
    unittest.main()

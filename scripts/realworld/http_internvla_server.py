import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from flask import Flask, jsonify, request
from PIL import Image
from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
from internnav.utils.instruction_task import InstructionTask as Task

CAMERA_POSE = np.eye(4)
CAMERA_INTRINSIC = np.array(
    [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def create_app(agent, task):
    app = Flask(__name__)
    started = False

    @app.route('/eval_dual', methods=['POST'])
    def eval_dual():
        nonlocal started
        now = time.monotonic()
        if json.loads(request.form['json'])['reset'] or not started:
            task.reset(now)
            agent.reset()
            started = True

        if task.deadline is not None and now >= task.deadline:
            task.index = len(task.steps)
        if task.completed:
            return jsonify(discrete_action=[0], task_completed=True)

        timed_out = task.step_deadline is not None and now >= task.step_deadline
        if timed_out:
            task.advance(now)
            agent.reset(preserve_memory=True)
            if task.completed:
                return jsonify(discrete_action=[0], task_completed=True)

        image = np.asarray(Image.open(request.files['image'].stream).convert('RGB'))
        depth = np.asarray(Image.open(request.files['depth'].stream).convert('I'), dtype=np.float32) / 10000.0
        inputs = (image, depth, CAMERA_POSE, task.instruction(), CAMERA_INTRINSIC)
        result = agent.step(*inputs)
        if result.output_action == [0]:
            # This STOP belongs to the instruction evaluated after any timeout transition.
            task.advance(time.monotonic())
            if task.completed:
                return jsonify(discrete_action=[0], task_completed=True)
            agent.reset(preserve_memory=True)
            return jsonify(discrete_action=[0])

        response = {'trajectory': result.output_trajectory.tolist()}
        if result.output_pixel is not None:
            response['pixel_goal'] = result.output_pixel
        return jsonify(response)

    return app


def main():
    # If the task keep interrupt, larger task time in the yaml file.
    parser = argparse.ArgumentParser()
    cache = Path(os.environ.get('MODELSCOPE_CACHE', 'checkpoints'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model_path', default=str(cache / 'InternVLA-N1'))
    parser.add_argument('--resize_w', type=int, default=384)
    parser.add_argument('--resize_h', type=int, default=384)
    parser.add_argument('--num_history', type=int, default=8)
    parser.add_argument('--plan_step_gap', type=int, default=4)
    parser.add_argument('--port', type=int, default=5801)
    parser.add_argument('--instruction_file', type=Path, default=Path(__file__).resolve().parents[2] / 'deploy/instructions.yaml')
    parser.add_argument('--task_name', help='Task in the YAML; defaults to the first one')
    args = parser.parse_args()

    tasks = yaml.safe_load(args.instruction_file.read_text(encoding='utf-8'))
    name = args.task_name or next(iter(tasks))
    if not tasks:
        parser.error(f'no tasks in {args.instruction_file}')
    if name not in tasks:
        parser.error(f'unknown task {name!r}; choose from: {", ".join(tasks)}')

    task = Task(name, tasks[name])
    print(f'Loaded task {name!r} ({len(task.steps)} steps)')

    # One agent and one task state are shared by the route.
    create_app(InternVLAN1AsyncAgent(args), task).run(host='0.0.0.0', port=args.port, threaded=False)


if __name__ == '__main__':
    main()

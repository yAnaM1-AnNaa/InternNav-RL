import argparse
import importlib.util
import sys
from pathlib import Path

import yaml
from real_world_env import RealWorldEnv
from stream import run, set_instruction

from internnav.utils.comm_utils.client import AgentClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='challenge/onsite_competition/configs/h1_internvla_n1_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        "--instruction_file",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "data" / "eval_instructions.yaml",
        help="YAML file with tasks and instructions",
    )
    parser.add_argument("--task_name", type=str, default=None, help="only run this task")
    parser.add_argument("--uninteractive_mode", action='store_true', help="whether to confirm each step")
    parser.add_argument("--max_steps", type=int, default=200, help="max steps per instruction")
    return parser.parse_args()


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


def confirm(msg: str) -> bool:
    """
    Ask user to confirm. Return True if user types 'y' (case-insensitive),
    False for anything else (including empty input).
    """
    try:
        answer = input(f"{msg} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return False
    return answer in ("", "y")


def load_instructions(path, task_name=None):
    tasks = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if task_name is not None:
        if task_name not in tasks:
            sys.exit(f"unknown task {task_name!r}; available: {', '.join(tasks)}")
        tasks = {task_name: tasks[task_name]}
    for name, task in tasks.items():
        for step_name, text in task['instructions'].items():
            yield f'{name}/{step_name}', text


def action_to_word(action: int) -> str:
    action = max(0, min(3, action))
    wl = ["stand still", "move forward", "turn left", "turn right"]
    return wl[action]


def main():
    args = parse_args()
    print("--- Loading config from:", args.config, "---")
    cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    agent_cfg = cfg.agent

    # initialize user's agent
    agent = AgentClient(agent_cfg)

    # initialize real world env
    env = RealWorldEnv(fps=30, duration=0.1, distance=0.3, angle=15, move_speed=0.5, turn_speed=0.5)
    env.reverse()  # reverse move direction if using a rear camera
    env.step(0)

    # start stream
    print("--- start running steam app ---")
    run(env=env)

    for label, instruction in load_instructions(args.instruction_file, args.task_name):
        print(f"\nNew instruction [{label}]:", instruction)
        set_instruction(instruction)
        agent.reset()

        for _ in range(args.max_steps):
            # obs contains {rgb, depth, instruction}
            obs = env.get_observation()
            obs["instruction"] = instruction

            # action is an integer in [0, 3]; 0 means STOP/stand still.
            action = agent.step([obs])[0]['action'][0]
            print("agent step success, action:", action)

            if args.uninteractive_mode or confirm(f"Execute this action [{action_to_word(action)}]?"):
                env.step(action)
                if action == 0:
                    break
            else:
                print("Stop requested. Exiting loop.")
                agent.reset()
                break


if __name__ == "__main__":
    main()

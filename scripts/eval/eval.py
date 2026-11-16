import sys

sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')

# Bootstrap the Isaac Sim / Omniverse Kit kernel first, so that `carb`/`omni`
# are importable when InternUtopia's import_extensions() runs below. In the
# official NGC container this is done by `source python.env.init`; the pip
# install needs an explicit import. Requires OMNI_KIT_ACCEPT_EULA=YES.
import isaacsim  # noqa: F401

import argparse
import importlib.util
from pathlib import Path

import yaml

from internnav.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_rdp_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument('--instruction_file', type=Path, help='YAML tasks to evaluate in Isaac Sim')
    parser.add_argument('--task_name', help='Run one YAML task by name, or all tasks when omitted')
    return parser.parse_args()


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


def main():
    args = parse_args()
    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')

    instruction_file = args.instruction_file or evaluator_cfg.eval_settings.get('instruction_file')
    if instruction_file is not None:
        if evaluator_cfg.eval_type != 'vln_distributed':
            raise ValueError('YAML instruction tasks require the Isaac Sim vln_distributed evaluator')
        tasks = yaml.safe_load(Path(instruction_file).read_text(encoding='utf-8'))
        if not tasks:
            raise ValueError(f'No tasks in {instruction_file}')
        if args.task_name is not None:
            tasks = {args.task_name: tasks[args.task_name]}
        evaluator_cfg.dataset.dataset_settings['instruction_tasks'] = tasks
    elif args.task_name is not None:
        raise ValueError('--task_name requires --instruction_file')

    # fill in evaluator default config
    if evaluator_cfg.eval_type == 'vln_distributed':
        from internnav.configs.evaluator.vln_default_config import get_config

        evaluator_cfg = get_config(evaluator_cfg)

    # create evaluator based on sim backend and run eval
    evaluator = Evaluator.init(evaluator_cfg)
    evaluator.eval()


if __name__ == '__main__':
    main()

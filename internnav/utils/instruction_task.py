class InstructionTask:
    """Run ordered instruction steps within an optional wall-clock budget."""

    def __init__(self, name, config):
        self.name = name
        self.steps = list(config['instructions'].items())
        self.total_time = config.get('total_time')
        self.index = 0
        self.deadline = None
        self.step_deadline = None

    @property
    def completed(self):
        return self.index == len(self.steps)

    def reset(self, now):
        self.index = 0
        self.deadline = None if self.total_time is None else now + self.total_time
        self.set_step_deadline(now)

    def set_step_deadline(self, now):
        self.step_deadline = None
        if self.deadline is not None and not self.completed:
            # Divide the remaining budget evenly among the remaining steps.
            self.step_deadline = now + (self.deadline - now) / (len(self.steps) - self.index)

    def advance(self, now):
        self.index += 1
        self.set_step_deadline(now)

    def instruction(self):
        name, text = self.steps[self.index]
        parts = '、'.join(key for key, _ in self.steps)
        return (
            f'Task name {self.name}, contains {parts}. '
            f'Now executing {self.index + 1} part ({name}), the specific instruction is {text}'
        )

import threading


class LatestLatentPlanner:
    """One System 2 worker, one active plan and one replaceable waiting plan.

    A consumer takes a snapshot before System 1 inference. Publishing never
    changes that snapshot. STOP invalidates both plans immediately.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._worker = None
        self._busy = False
        self._generation = 0
        self._active = None
        self._pending = None
        self._stopped = False
        self._error = None

    @property
    def busy(self):
        with self._condition:
            return self._busy

    def submit(self, infer, *args):
        with self._condition:
            if self._busy or self._stopped or self._error is not None:
                return False
            self._busy = True
            generation = self._generation
            self._worker = threading.Thread(
                target=self._run, args=(generation, infer, args), daemon=True
            )
            self._worker.start()
            return True

    def _run(self, generation, infer, args):
        plan = None
        failure = None
        try:
            plan = infer(*args)
        except Exception as error:
            failure = error
        with self._condition:
            try:
                if generation == self._generation:
                    if failure is None:
                        self.publish(generation, plan)
                    else:
                        self._error = failure
            except Exception as error:
                self._error = error
            finally:
                self._busy = False
                self._condition.notify_all()

    def publish(self, generation, plan):
        with self._condition:
            if generation != self._generation or self._stopped:
                return
            if plan.output_action == [0]:
                self._stopped = True
                self._active = self._pending = None
            else:
                if plan.output_latent is None:
                    raise ValueError('System 2 returned neither STOP nor a latent plan')
                self._pending = plan
            self._condition.notify_all()

    def take(self):
        """Wait only for the first plan; otherwise consume latest or reuse active."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._stopped
                    or self._error is not None
                    or self._pending is not None
                    or self._active is not None
                )
            )
            if self._error is not None:
                error, self._error = self._error, None
                raise RuntimeError('System 2 inference failed') from error
            if self._stopped:
                return self._generation, None
            if self._pending is not None:
                self._active, self._pending = self._pending, None
            return self._generation, self._active

    def is_current(self, generation):
        with self._condition:
            return generation == self._generation and not self._stopped

    def reset(self):
        # Invalidate before joining so a late result cannot become a new plan.
        # Join before the caller resets the VLM's recurrent memory/history.
        with self._condition:
            self._generation += 1
            self._active = self._pending = None
            self._stopped = False
            self._error = None
            worker = self._worker
        if worker is not None:
            worker.join()

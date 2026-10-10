"""The Space's model pass (serving/space_app.py): a ZeroGPU GPU when one is granted, else the CPU."""
import time


class GpuOrCpu:
    """
    Calls gpu(x); when it raises (ZeroGPU allowance used up, queue too long), answers with cpu(x) instead and skips
    the GPU for `cooldown` seconds, so a refused GPU doesn't slow every request. `.device` says where photos run now.
    """

    def __init__(self, gpu, cpu, cooldown=600, clock=time.monotonic):
        self.gpu, self.cpu, self.cooldown, self.clock = gpu, cpu, cooldown, clock
        self.cpu_until = float("-inf")

    @property
    def device(self):
        return "zerogpu" if self.clock() >= self.cpu_until else "cpu (ZeroGPU refused a recent call)"

    def __call__(self, x):
        if self.clock() >= self.cpu_until:
            try:
                return self.gpu(x)
            except Exception as error:  # ZeroGPU raises gradio.Error for quota and queue refusals
                self.cpu_until = self.clock() + self.cooldown
                print(f"ZeroGPU refused, CPU for {self.cooldown} s: {type(error).__name__}: {error}", flush=True)
        return self.cpu(x)

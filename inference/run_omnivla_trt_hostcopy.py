import os
import runpy
from pathlib import Path


# Force the legacy TensorRT pybind path:
# torch CUDA tensor -> CPU numpy -> TRT GPU -> CPU numpy -> torch CUDA tensor
os.environ["OMNIVLA_TRTLLM_FORCE_HOST_COPY"] = "1"

runpy.run_path(str(Path(__file__).with_name("run_omnivla.py")), run_name="__main__")

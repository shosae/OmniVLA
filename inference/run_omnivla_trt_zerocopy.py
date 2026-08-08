import os
import runpy
from pathlib import Path


# Force the direct CUDA-pointer TensorRT pybind path when available.
os.environ["OMNIVLA_TRTLLM_FORCE_HOST_COPY"] = "0"

runpy.run_path(str(Path(__file__).with_name("run_omnivla.py")), run_name="__main__")

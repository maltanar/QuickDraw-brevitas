#!/bin/bash

# This script launches the experiments inside the custom quickdraw-qat image.
# It assumes you have an AMD GPU and ROCm drivers installed.

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v "$(pwd):/workspace/quickdraw-brevitas" \
  -w /workspace/quickdraw-brevitas \
  quickdraw-qat \
  bash ./run_experiments.sh

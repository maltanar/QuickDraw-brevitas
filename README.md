# QuickDraw-brevitas

Train Quick, Draw! classifiers with PyTorch and run quantization-aware training (QAT) with Brevitas. The repo is installable as a standard Python project in a virtual environment, or inside Docker.

## Quick Start (venv)

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install the project with all dependencies in editable mode.

```bash
pip install -e ".[full]"
```

## Available CLI Commands

Installing the project creates these commands:

1. `quickdraw-prepare-data`
2. `quickdraw-train`
3. `quickdraw-qat-train`
4. `quickdraw-convert-qdq-to-qop`

## Typical Local Workflow

1. Prepare dataset:

```bash
quickdraw-prepare-data --download 1 --categories 10 -v 0.2
```

2. Run baseline (floating-point, should get ~95% accuracy) training for 5 epochs, and export to ONNX:

```bash
quickdraw-train --ngpu 0 -e 15 --export_onnx
```

3. Run QAT (8-bit quantization, should get ~95% accuracy) for 15 epochs, and export to QCDQ-formatted ONNX:

```bash
quickdraw-qat-train --ngpu 0 -e 15 --weight_bit_width 8 --act_bit_width 8 --quant_input --no_narrow_range  --export_qcdq
```

4. Convert QCDQ ONNX to QOperator ONNX:

```bash
quickdraw-convert-qdq-to-qop Checkpoints/model_8bit_qcdq.onnx Checkpoints/model_8bit_qop.onnx
```

## Existing Script Workflow

The shell helpers still work:

```bash
bash run_experiments.sh
bash run_2bit_experiments.sh
```

## Docker Workflow

The Docker image now installs this project via `pyproject.toml` and the `full` extra.

1. Build image:

```bash
docker build -t quickdraw-qat .
```

2. Run containerized experiments:

```bash
bash launch_docker.sh
```

## Dataset Notes

[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) contains 50M drawings across 345 categories. This project typically samples up to 5000 examples per class for training data generation.

Useful data-prep options:

1. `--categories` / `-c`: one of `10`, `30`, `100`, `all`
2. `--download` / `-d`: `1` to download raw `.npy` files, `0` to reuse local files
3. `--show_random_imgs` / `-show`: preview random generated samples

### Files produced by quickdraw-prepare-data

Running the command below:

```bash
quickdraw-prepare-data --download 1 --categories 10 -v 0.2
```

produces or updates these artifacts:

1. Raw class files in [Data](Data)
	- One file per class, for example [Data/eye.npy](Data/eye.npy) and [Data/car.npy](Data/car.npy).

2. Training split in [Dataset/train.npz](Dataset/train.npz)
	- Keys: `data` and `target`.
	- `data` shape is `(N_train, 784)` where each row is a flattened 28x28 grayscale sketch.
	- `target` shape is `(N_train,)` with integer class ids.

3. Test split in [Dataset/test.npz](Dataset/test.npz)
	- Keys: `data` and `target`.
	- Same format as training split.

4. Class mapping metadata in [Dataset/class_names.txt](Dataset/class_names.txt)
	- One line per class with class name and sample count.
	- This file is used by the loader to recover the number of classes and label mapping context.

For your sample run (`10` classes, `5000` max samples per class, validation fold `0.2`):

1. Total sampled examples: `10 * 5000 = 50000`
2. Test examples: `50000 * 0.2 = 10000`
3. Train examples: `50000 - 10000 = 40000`

That matches the printed output:

1. `x_train size: (40000, 784)`
2. `x_test size: (10000, 784)`

Note: integer class ids are assigned in the order files are loaded during dataset generation, so [Dataset/class_names.txt](Dataset/class_names.txt) is the source of truth for id-to-class interpretation.

## References

1. [Train a model in tf.keras with Colab, and run it in the browser with TensorFlow.js](https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e)
2. [tfjs-converter](https://github.com/tensorflow/tfjs-converter)
3. [pytorch2keras](https://github.com/nerox8664/pytorch2keras)



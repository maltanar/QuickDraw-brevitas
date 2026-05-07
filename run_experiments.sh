#!/bin/bash

echo "Preparing data..."
python DataUtils/prepare_data.py --download 1 --categories 10 -v 0.2

echo "Running 8-bit QAT..."
python main_qat.py --bit_width 8 --quant_input --epochs 15 --export_qonnx

#echo "Running 4-bit QAT..."
#python main_qat.py --bit_width 4 --epochs 25 --export_qonnx --no_narrow_range --ngpu 0

#echo "Running 2-bit QAT..."
#python main_qat.py --bit_width 2 --epochs 100 --export_qonnx

echo "All experiments complete."

#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <model-name>" >&2
	exit 1
fi

CC="${CC:-arm-zephyr-eabi-gcc}"
MODEL_NAME="$1"
INPUT_PATH="Checkpoints/${MODEL_NAME}.onnx"
OUTPUT_DIR="codegen/"
MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
MODEL_C_PATH="$MODEL_DIR/model.c"
MODEL_BIN_PATH="$MODEL_DIR/model"

to_human_bytes() {
	local bytes="$1"
	if command -v numfmt >/dev/null 2>&1; then
		numfmt --to=iec-i --suffix=B "$bytes"
	else
		echo "${bytes} B"
	fi
}

sum_sections_by_prefix() {
	local binary="$1"
	local section_regex="$2"

	size -A -d "$binary" \
		| awk -v rx="$section_regex" '$1 ~ rx { sum += $2 } END { print sum + 0 }'
}

extract_elf_gnu_stack() {
	local binary="$1"
	if ! command -v readelf >/dev/null 2>&1; then
		echo "N/A (readelf not found)"
		return 0
	fi

	local stack_line
	stack_line="$(readelf -W -l "$binary" | awk '$1 == "GNU_STACK" { print; exit }')"

	if [ -z "$stack_line" ]; then
		echo "N/A (no GNU_STACK program header)"
		return 0
	fi

	local memsz_hex flags
	memsz_hex="$(echo "$stack_line" | awk '{ print $6 }')"
	flags="$(echo "$stack_line" | awk '{ print $7 }')"

	if [ "$memsz_hex" = "0x0" ] || [ "$memsz_hex" = "0x000000" ] || [ "$memsz_hex" = "0x0000000000000000" ]; then
		echo "requested_size=0 (OS/runtime decides), flags=${flags}"
	else
		echo "requested_size=${memsz_hex}, flags=${flags}"
	fi
}

mkdir -p "$MODEL_DIR"
# --large-temp-threshold 1024000
emx-onnx-cgen compile --large-weight-threshold 0 --emit-testbench "$INPUT_PATH" "$MODEL_C_PATH"
$CC -Os -std=c23 "$MODEL_C_PATH" -c -o "$MODEL_BIN_PATH" -lm

if ! command -v size >/dev/null 2>&1; then
	echo "Warning: 'size' command not found; skipping binary section analysis." >&2
	exit 0
fi

FILE_SIZE_BYTES="$(stat -c%s "$MODEL_BIN_PATH")"
TEXT_SIZE_BYTES="$(sum_sections_by_prefix "$MODEL_BIN_PATH" '^\\.text($|\\.)|^\\.init$|^\\.fini$|^\\.plt($|\\.)')"
RODATA_SIZE_BYTES="$(sum_sections_by_prefix "$MODEL_BIN_PATH" '^\\.rodata($|\\.)')"
DATA_SIZE_BYTES="$(sum_sections_by_prefix "$MODEL_BIN_PATH" '^\\.data($|\\.)')"
BSS_SIZE_BYTES="$(sum_sections_by_prefix "$MODEL_BIN_PATH" '^\\.bss($|\\.)')"
STATIC_RAM_BYTES="$((DATA_SIZE_BYTES + BSS_SIZE_BYTES))"
FLASH_IMAGE_BYTES="$((TEXT_SIZE_BYTES + RODATA_SIZE_BYTES + DATA_SIZE_BYTES))"
GNU_STACK_INFO="$(extract_elf_gnu_stack "$MODEL_BIN_PATH")"

RUNTIME_STACK_LIMIT_KB="$(ulimit -s 2>/dev/null || true)"
if [ -z "$RUNTIME_STACK_LIMIT_KB" ]; then
	RUNTIME_STACK_LIMIT_DISPLAY="N/A"
elif [ "$RUNTIME_STACK_LIMIT_KB" = "unlimited" ]; then
	RUNTIME_STACK_LIMIT_DISPLAY="unlimited"
else
	RUNTIME_STACK_LIMIT_DISPLAY="${RUNTIME_STACK_LIMIT_KB} KiB"
fi

echo
echo "=== Executable Deployment Stats: ${MODEL_BIN_PATH} ==="
echo "File size:                   ${FILE_SIZE_BYTES} bytes ($(to_human_bytes "$FILE_SIZE_BYTES"))"
echo "Code segment (.text*):       ${TEXT_SIZE_BYTES} bytes ($(to_human_bytes "$TEXT_SIZE_BYTES"))"
echo "Read-only vars (.rodata*):   ${RODATA_SIZE_BYTES} bytes ($(to_human_bytes "$RODATA_SIZE_BYTES"))"
echo "Writable init data (.data*): ${DATA_SIZE_BYTES} bytes ($(to_human_bytes "$DATA_SIZE_BYTES"))"
echo "Zero-init data (.bss*):      ${BSS_SIZE_BYTES} bytes ($(to_human_bytes "$BSS_SIZE_BYTES"))"
echo "Static RAM (.data + .bss):   ${STATIC_RAM_BYTES} bytes ($(to_human_bytes "$STATIC_RAM_BYTES"))"
echo "Flash image estimate:        ${FLASH_IMAGE_BYTES} bytes ($(to_human_bytes "$FLASH_IMAGE_BYTES"))"
echo "ELF GNU_STACK:               ${GNU_STACK_INFO}"
echo "Runtime stack limit:         ${RUNTIME_STACK_LIMIT_DISPLAY}"


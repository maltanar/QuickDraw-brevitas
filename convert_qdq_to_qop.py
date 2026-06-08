import argparse
import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import SortGraph

def get_node_by_output(nodes, name):
    for node in nodes:
        if name in node.output:
            return node
    return None

def find_consumers(nodes, name):
    consumers = []
    for node in nodes:
        if name in node.input:
            consumers.append(node)
    return consumers

def get_initializer(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init
    return None

def get_tensor_dtype(graph, name):
    """Get the data type of a tensor from initializers or value_info."""
    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            return init.data_type
    
    # Check value_info
    for vi in graph.value_info:
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    
    # Check graph inputs
    for inp in graph.input:
        if inp.name == name:
            return inp.type.tensor_type.elem_type
    
    return None

def is_int8_or_uint8(dtype):
    """Check if dtype is int8 or uint8 in ONNX."""
    return dtype in (onnx.TensorProto.UINT8, onnx.TensorProto.INT8)


def preprocess_brevitas_fc_patterns(graph):
    """Preprocess Brevitas FC quantized weights: Constant -> Cast -> DQ -> Transpose -> MatMul

    Pre-transpose the weights in the initializer and remove the Transpose node from the graph.
    This allows the standard MatMul pattern matching to handle the conversion.

    Returns: number of patterns preprocessed
    """
    transpose_outputs_to_remove = set()
    initializers_to_update = {}
    preprocessed = 0

    for matmul_node in list(graph.node):
        if matmul_node.op_type != 'MatMul':
            continue

        # Check both inputs for Transpose
        for input_idx in range(2):
            input_name = matmul_node.input[input_idx]
            transpose_node = get_node_by_output(graph.node, input_name)

            if not transpose_node or transpose_node.op_type != 'Transpose':
                continue

            # Trace: Transpose -> DequantizeLinear
            transpose_input = transpose_node.input[0]
            dq_node = get_node_by_output(graph.node, transpose_input)

            if not dq_node or dq_node.op_type != 'DequantizeLinear':
                continue

            # Trace: DequantizeLinear -> Cast (optional) -> Constant
            dq_input = dq_node.input[0]
            cast_node = get_node_by_output(graph.node, dq_input)

            # Cast is optional - check if it exists, otherwise use dq_input directly
            if cast_node and cast_node.op_type == 'Cast':
                const_name = cast_node.input[0]
            else:
                const_name = dq_input

            const_init = get_initializer(graph, const_name)

            if not const_init:
                continue

            # Found the pattern! Pre-transpose the weights
            arr = numpy_helper.to_array(const_init)

            # Get transpose axes
            axes = None
            for attr in transpose_node.attribute:
                if attr.name == 'perm':
                    axes = list(helper.get_attribute_value(attr))

            # Transpose the array
            if axes is not None:
                arr = np.transpose(arr, axes)
            else:
                arr = np.transpose(arr)

            # Create new transposed initializer
            transposed_init = numpy_helper.from_array(arr, const_name)
            initializers_to_update[const_name] = transposed_init

            # Rewire MatMul to skip Transpose and use the DQ output directly
            matmul_node.input[input_idx] = transpose_input

            # Mark Transpose node for removal
            transpose_outputs_to_remove.add(transpose_node.output[0])

            preprocessed += 1
            print(f"Preprocessed Brevitas FC pattern in {matmul_node.name}: pre-transposed weights and removed Transpose")

    # Update initializers with transposed versions
    for old_name, new_init in initializers_to_update.items():
        for i, init in enumerate(graph.initializer):
            if init.name == old_name:
                graph.initializer[i].CopyFrom(new_init)
                break

    # Remove Transpose nodes from graph
    if transpose_outputs_to_remove:
        kept_nodes = []
        for node in graph.node:
            if node.op_type == 'Transpose' and node.output and node.output[0] in transpose_outputs_to_remove:
                continue
            kept_nodes.append(node)
        del graph.node[:]
        graph.node.extend(kept_nodes)

    return preprocessed


def _get_clip_bounds(graph, clip_node):
    """Return scalar (min, max) for a Clip node, or (None, None) if unavailable."""
    # Clip-11+ typically uses inputs for min/max.
    if len(clip_node.input) >= 3:
        min_init = get_initializer(graph, clip_node.input[1])
        max_init = get_initializer(graph, clip_node.input[2])
        if min_init is not None and max_init is not None:
            min_val = float(np.asarray(numpy_helper.to_array(min_init)).reshape(-1)[0])
            max_val = float(np.asarray(numpy_helper.to_array(max_init)).reshape(-1)[0])
            return min_val, max_val

    # Fallback for older Clip variants with attributes.
    min_val = None
    max_val = None
    for attr in clip_node.attribute:
        if attr.name == "min":
            min_val = helper.get_attribute_value(attr)
        elif attr.name == "max":
            max_val = helper.get_attribute_value(attr)
    if min_val is not None and max_val is not None:
        return float(min_val), float(max_val)

    return None, None


def _infer_lowbit_qdtype(min_val, max_val):
    """Infer low-bit quantized type from Clip range.

    Returns (q_dtype, cast_dtype, suffix) or (None, None, None).
    """
    if min_val is None or max_val is None:
        return None, None, None

    mn = int(round(min_val))
    mx = int(round(max_val))

    # Signed 4-bit full range and narrow range.
    if (mn, mx) in {(-8, 7), (-7, 7)}:
        return onnx.TensorProto.INT4, onnx.TensorProto.INT8, "int4"

    # Unsigned 4-bit full range and narrow range.
    if (mn, mx) in {(0, 15), (0, 14)}:
        return onnx.TensorProto.UINT4, onnx.TensorProto.UINT8, "uint4"

    return None, None, None


def rewrite_qcdq_lowbit_patterns(graph):
    """Rewrite QuantizeLinear->Clip->DequantizeLinear into low-bit Q + Cast.

    Pattern:
    QuantizeLinear -> Clip -> DequantizeLinear

    Rewritten as:
    QuantizeLinear(low-bit output type) -> Cast(to int8/uint8) -> DequantizeLinear

    This keeps DequantizeLinear input in 8-bit while preserving low-bit intent.
    """
    added_nodes = []
    added_initializers = []
    clip_outputs_to_remove = set()
    processed_clip_outputs = set()
    rewrites = 0

    # Iterate over DQ nodes and match upstream Clip <- QuantizeLinear.
    for dq_node in list(graph.node):
        if dq_node.op_type != "DequantizeLinear":
            continue

        clip_node = get_node_by_output(graph.node, dq_node.input[0])
        if clip_node is None or clip_node.op_type != "Clip":
            continue
        if clip_node.output and clip_node.output[0] in processed_clip_outputs:
            continue
        if len(clip_node.input) < 1:
            continue

        min_val, max_val = _get_clip_bounds(graph, clip_node)
        q_dtype, cast_dtype, dtype_suffix = _infer_lowbit_qdtype(min_val, max_val)
        if q_dtype is None:
            continue

        q_node = get_node_by_output(graph.node, clip_node.input[0])

        # Case 1: QuantizeLinear -> Clip -> DequantizeLinear
        if q_node is not None and q_node.op_type == "QuantizeLinear":
            # Ensure QuantizeLinear has an explicit zero-point so output dtype is well-defined.
            if len(q_node.input) < 3 or q_node.input[2] == "":
                zp_base_name = q_node.name + "_zp_base"
                if cast_dtype == onnx.TensorProto.INT8:
                    zp_base_init = numpy_helper.from_array(np.array(0, dtype=np.int8), zp_base_name)
                else:
                    zp_base_init = numpy_helper.from_array(np.array(0, dtype=np.uint8), zp_base_name)
                added_initializers.append(zp_base_init)
                q_node.input.append(zp_base_name)

            # Cast zero-point to low-bit dtype and feed it to QuantizeLinear.
            old_zp_name = q_node.input[2]
            new_zp_name = q_node.name + "_zp_" + dtype_suffix
            zp_cast_name = q_node.name + "_zp_to_" + dtype_suffix
            zp_cast_node = helper.make_node(
                "Cast",
                inputs=[old_zp_name],
                outputs=[new_zp_name],
                name=zp_cast_name,
                to=q_dtype,
            )
            added_nodes.append(zp_cast_node)
            q_node.input[2] = new_zp_name

            # Rewire Q output through an explicit Cast back to 8-bit where Clip output used to be.
            old_q_output = q_node.output[0]
            lowbit_q_output = old_q_output + "_" + dtype_suffix
            q_node.output[0] = lowbit_q_output

            to_8bit_cast = helper.make_node(
                "Cast",
                inputs=[lowbit_q_output],
                outputs=[clip_node.output[0]],
                name=q_node.name + "_" + dtype_suffix + "_to_8bit",
                to=cast_dtype,
            )
            added_nodes.append(to_8bit_cast)

            clip_outputs_to_remove.add(clip_node.output[0])
            processed_clip_outputs.add(clip_node.output[0])
            rewrites += 1
            continue

        # Case 2: Constant(int8/uint8 initializer) -> Clip -> DequantizeLinear
        src_init = get_initializer(graph, clip_node.input[0])
        if src_init is None:
            continue
        if src_init.data_type not in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8):
            continue

        src_val = numpy_helper.to_array(src_init)
        clipped = np.clip(src_val, min_val, max_val)

        # Create compact low-bit initializer and cast it back to 8-bit at the old Clip output.
        if q_dtype == onnx.TensorProto.INT4:
            clipped_vals = np.rint(clipped).astype(np.int8).reshape(-1).tolist()
        else:
            clipped_vals = np.rint(clipped).astype(np.uint8).reshape(-1).tolist()

        if clip_node.name:
            base_name = clip_node.name
        else:
            base_name = clip_node.output[0].replace("/", "_")
        lowbit_init_name = base_name + "_const_" + dtype_suffix
        lowbit_init = helper.make_tensor(
            name=lowbit_init_name,
            data_type=q_dtype,
            dims=list(src_val.shape),
            vals=clipped_vals,
        )
        added_initializers.append(lowbit_init)

        const_cast_node = helper.make_node(
            "Cast",
            inputs=[lowbit_init_name],
            outputs=[clip_node.output[0]],
            name=base_name + "_" + dtype_suffix + "_to_8bit",
            to=cast_dtype,
        )
        added_nodes.append(const_cast_node)

        clip_outputs_to_remove.add(clip_node.output[0])
        processed_clip_outputs.add(clip_node.output[0])
        rewrites += 1

    if rewrites == 0:
        return 0

    # Drop rewritten Clip nodes and append created Cast nodes.
    kept_nodes = []
    for node in graph.node:
        if node.op_type == "Clip" and node.output and node.output[0] in clip_outputs_to_remove:
            continue
        kept_nodes.append(node)

    kept_nodes.extend(added_nodes)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    graph.initializer.extend(added_initializers)

    return rewrites

def convert_qdq_to_qop(model_path, output_path):
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph

    rewritten_qcdq = rewrite_qcdq_lowbit_patterns(graph)
    if rewritten_qcdq > 0:
        print(f"Rewrote {rewritten_qcdq} low-bit Clip pattern(s) to compact low-bit tensors + Cast.")

    preprocessed_fc = preprocess_brevitas_fc_patterns(graph)
    if preprocessed_fc > 0:
        print(f"Preprocessed {preprocessed_fc} Brevitas FC pattern(s): pre-transposed weights and removed Transpose node(s).")

    new_nodes = []
    outputs_to_remove = set()
    
    new_initializers = []

    # Find all Conv nodes
    for node in graph.node:
        if node.op_type == 'Conv':
            # Check if inputs are DequantizeLinear
            x_name = node.input[0]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else ""

            x_dq = get_node_by_output(graph.node, x_name)
            w_dq = get_node_by_output(graph.node, w_name)
            
            if not (x_dq and w_dq and x_dq.op_type == 'DequantizeLinear' and w_dq.op_type == 'DequantizeLinear'):
                print(f"Skipping {node.name}: inputs are not DequantizeLinear. x_dq: {x_dq.op_type if x_dq else 'None'}, w_dq: {w_dq.op_type if w_dq else 'None'}")
                continue
            
            # Check if output is consumed by QuantizeLinear
            y_name = node.output[0]
            consumers = find_consumers(graph.node, y_name)
            
            # Get the quantized inputs
            x_q_name = x_dq.input[0]
            x_scale_name = x_dq.input[1]
            x_zp = x_dq.input[2] if len(x_dq.input) > 2 else ""
            
            w_q_name = w_dq.input[0]
            w_scale_name = w_dq.input[1]
            w_zp = w_dq.input[2] if len(w_dq.input) > 2 else ""

            if len(consumers) == 1 and consumers[0].op_type == 'QuantizeLinear':
                # DequantizeLinear (x2) -> Conv -> QuantizeLinear  =>  QLinearConv
                # Check if quantized tensors are int8/uint8 (QLinearConv doesn't support sub-8-bit)
                x_dtype = get_tensor_dtype(graph, x_q_name)
                w_dtype = get_tensor_dtype(graph, w_q_name)
                
                if x_dtype is not None and w_dtype is not None and is_int8_or_uint8(x_dtype) and is_int8_or_uint8(w_dtype):
                    y_q = consumers[0]
                    y_scale = y_q.input[1]
                    y_zp = y_q.input[2] if len(y_q.input) > 2 else ""
                    q_conv_output = y_q.output[0]

                    inputs = [
                        x_q_name, x_scale_name, x_zp,
                        w_q_name, w_scale_name, w_zp,
                        y_scale, y_zp
                    ]
                    if b_name:
                        # Quantize bias to int32
                        b_init = get_initializer(graph, b_name)
                        xs_init = get_initializer(graph, x_scale_name)
                        ws_init = get_initializer(graph, w_scale_name)
                        
                        if b_init and xs_init and ws_init:
                            b_val = numpy_helper.to_array(b_init)
                            xs_val = numpy_helper.to_array(xs_init)
                            ws_val = numpy_helper.to_array(ws_init)
                            
                            # QLinearConv expects bias to be 1D [C_out]
                            bq_val = np.round(b_val.flatten() / (xs_val * ws_val)).astype(np.int32)
                            bq_name = b_name + "_quantized"
                            bq_init = numpy_helper.from_array(bq_val, bq_name)
                            new_initializers.append(bq_init)
                            inputs.append(bq_name)
                        else:
                            print(f"Warning: Could not quantize bias for {node.name}, using original bias name. This may fail.")
                            inputs.append(b_name)

                    new_node = helper.make_node(
                        'QLinearConv',
                        inputs=inputs,
                        outputs=[q_conv_output],
                        name=node.name + "_quant",
                        **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                    )
                    new_nodes.append(new_node)
                    outputs_to_remove.add(node.output[0])
                    outputs_to_remove.add(x_dq.output[0])
                    outputs_to_remove.add(w_dq.output[0])
                    outputs_to_remove.add(y_q.output[0])
                    print(f"Replaced {node.name} (Conv) with QLinearConv")
                else:
                    # Quantized tensors are not int8/uint8, skip conversion
                    print(f"Skipping {node.name}: quantized tensors are not int8/uint8 (x_dtype: {x_dtype}, w_dtype: {w_dtype}). Sub-8-bit types not supported by QLinearConv.")
                    continue

            else:
                # DequantizeLinear (x2) -> Conv  (no following QuantizeLinear)  =>  ConvInteger + Scaling
                inputs = [x_q_name, w_q_name]
                if x_zp:
                    inputs.append(x_zp)
                if w_zp:
                    if not x_zp:
                        inputs.append("")
                    inputs.append(w_zp)

                conv_int_out = node.name + "_int_output"
                new_node = helper.make_node(
                    'ConvInteger',
                    inputs=inputs,
                    outputs=[conv_int_out],
                    name=node.name + "_integer",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                
                # Add DequantizeLinear to restore scale
                xs_init = get_initializer(graph, x_scale_name)
                ws_init = get_initializer(graph, w_scale_name)
                
                if xs_init and ws_init:
                    xs_val = numpy_helper.to_array(xs_init)
                    ws_val = numpy_helper.to_array(ws_init)
                    comb_s_val = xs_val * ws_val
                    comb_s_name = node.name + "_combined_scale"
                    comb_s_init = numpy_helper.from_array(comb_s_val.astype(np.float32), comb_s_name)
                    new_initializers.append(comb_s_init)
                    
                    dq_out = node.name + "_dq_output"
                    # If there's a bias, we'll add it after DQ
                    # If no bias, DQ output is the final node output
                    final_node_out = node.output[0] if not b_name else dq_out
                    
                    dq_node = helper.make_node(
                        'DequantizeLinear',
                        inputs=[conv_int_out, comb_s_name],
                        outputs=[final_node_out],
                        name=node.name + "_dq"
                    )
                    new_nodes.append(dq_node)
                    
                    if b_name:
                        # For Add to broadcast correctly [N, C, H, W] + [1, C, 1, 1]
                        # We need to reshape the bias if it's 1D
                        b_init = get_initializer(graph, b_name)
                        if b_init:
                            b_val = numpy_helper.to_array(b_init)
                            if b_val.ndim == 1:
                                # Get Conv attributes to determine rank
                                # Default is 2D conv (4D tensor)
                                # But we can check if it has 'strides' or other attrs
                                # For now assume 2D conv if not specified, or just use rank of conv output if we knew it
                                # Use the weight initializer shape to determine rank
                                # w_q_name was found earlier (w_dq.input[0])
                                w_init = get_initializer(graph, w_q_name)
                                if w_init:
                                    w_rank = len(w_init.dims)
                                    new_shape = [1] * w_rank
                                    new_shape[1] = b_val.shape[0]
                                    b_reshaped_val = b_val.reshape(new_shape)
                                    b_reshaped_name = b_name + "_reshaped"
                                    b_reshaped_init = numpy_helper.from_array(b_reshaped_val, b_reshaped_name)
                                    new_initializers.append(b_reshaped_init)
                                    bias_to_add = b_reshaped_name
                                else:
                                    bias_to_add = b_name
                            else:
                                bias_to_add = b_name
                        else:
                            bias_to_add = b_name

                        add_node = helper.make_node(
                            'Add',
                            inputs=[dq_out, bias_to_add],
                            outputs=[node.output[0]],
                            name=node.name + "_bias_add"
                        )
                        new_nodes.append(add_node)
                else:
                    print(f"Warning: Could not find scales for {node.name}, ConvInteger output will be unscaled!")
                    # Just rename output of ConvInteger to original output
                    new_node.output[0] = node.output[0]

                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                print(f"Replaced {node.name} (Conv) with ConvInteger + scaling")

        elif node.op_type == 'MatMul':
            x_name = node.input[0]
            w_name = node.input[1]
            
            x_dq = get_node_by_output(graph.node, x_name)
            w_dq = get_node_by_output(graph.node, w_name)
            
            if not (x_dq and w_dq and x_dq.op_type == 'DequantizeLinear' and w_dq.op_type == 'DequantizeLinear'):
                continue

            y_name = node.output[0]
            consumers = find_consumers(graph.node, y_name)

            # Get the quantized inputs
            x_q_name = x_dq.input[0]
            x_scale_name = x_dq.input[1]
            x_zp = x_dq.input[2] if len(x_dq.input) > 2 else ""

            w_q_name = w_dq.input[0]
            w_scale_name = w_dq.input[1]
            w_zp = w_dq.input[2] if len(w_dq.input) > 2 else ""

            if len(consumers) == 1 and consumers[0].op_type == 'QuantizeLinear':
                # DequantizeLinear (x2) -> MatMul -> QuantizeLinear  =>  QLinearMatMul
                # Check if quantized tensors are int8/uint8 (QLinearMatMul doesn't support sub-8-bit)
                x_dtype = get_tensor_dtype(graph, x_q_name)
                w_dtype = get_tensor_dtype(graph, w_q_name)
                
                if x_dtype is not None and w_dtype is not None and is_int8_or_uint8(x_dtype) and is_int8_or_uint8(w_dtype):
                    y_q = consumers[0]
                    y_scale = y_q.input[1]
                    y_zp = y_q.input[2] if len(y_q.input) > 2 else ""
                    q_matmul_output = y_q.output[0]

                    inputs = [
                        x_q_name, x_scale_name, x_zp,
                        w_q_name, w_scale_name, w_zp,
                        y_scale, y_zp
                    ]

                    new_node = helper.make_node(
                        'QLinearMatMul',
                        inputs=inputs,
                        outputs=[q_matmul_output],
                        name=node.name + "_quant",
                        **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                    )
                    new_nodes.append(new_node)
                    outputs_to_remove.add(node.output[0])
                    outputs_to_remove.add(x_dq.output[0])
                    outputs_to_remove.add(w_dq.output[0])
                    outputs_to_remove.add(y_q.output[0])
                    print(f"Replaced {node.name} (MatMul) with QLinearMatMul")
                else:
                    # Quantized tensors are not int8/uint8, skip conversion
                    print(f"Skipping {node.name}: quantized tensors are not int8/uint8 (x_dtype: {x_dtype}, w_dtype: {w_dtype}). Sub-8-bit types not supported by QLinearMatMul.")
                    continue

            else:
                # DequantizeLinear (x2) -> MatMul  (no following QuantizeLinear)  =>  MatMulInteger + Scaling
                inputs = [x_q_name, w_q_name]
                if x_zp:
                    inputs.append(x_zp)
                if w_zp:
                    if not x_zp:
                        inputs.append("")  # pad x_zp slot
                    inputs.append(w_zp)

                matmul_int_out = node.name + "_int_output"
                new_node = helper.make_node(
                    'MatMulInteger',
                    inputs=inputs,
                    outputs=[matmul_int_out],
                    name=node.name + "_integer",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                
                # Add DequantizeLinear to restore scale
                xs_init = get_initializer(graph, x_scale_name)
                ws_init = get_initializer(graph, w_scale_name)
                
                if xs_init and ws_init:
                    xs_val = numpy_helper.to_array(xs_init)
                    ws_val = numpy_helper.to_array(ws_init)
                    comb_s_val = xs_val * ws_val
                    comb_s_name = node.name + "_combined_scale"
                    comb_s_init = numpy_helper.from_array(comb_s_val.astype(np.float32), comb_s_name)
                    new_initializers.append(comb_s_init)
                    
                    dq_node = helper.make_node(
                        'DequantizeLinear',
                        inputs=[matmul_int_out, comb_s_name],
                        outputs=[node.output[0]],
                        name=node.name + "_dq"
                    )
                    new_nodes.append(dq_node)
                else:
                    print(f"Warning: Could not find scales for {node.name}, MatMulInteger output will be unscaled!")
                    new_node.output[0] = node.output[0]

                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                print(f"Replaced {node.name} (MatMul) with MatMulInteger + scaling")

    if not new_nodes:
        print("No QDQ patterns found to replace.")
        return
        
    final_nodes = []
    for node in graph.node:
        if node.output[0] not in outputs_to_remove:
            final_nodes.append(node)
            
    final_nodes.extend(new_nodes)
    
    # Create new graph
    new_initializers_total = list(graph.initializer) + new_initializers
    
    new_graph = helper.make_graph(
        final_nodes,
        graph.name,
        graph.input,
        graph.output,
        new_initializers_total,
        None,
        graph.value_info
    )
    
    new_model = helper.make_model(new_graph, producer_name='qdq-to-qop-converter', opset_imports=model.opset_import)
    sorted_model = ModelWrapper(new_model).transform(SortGraph())
    
    print(f"Saving QOp model to {output_path}...")
    sorted_model.save(output_path)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX QDQ format to QOp format")
    parser.add_argument("input_onnx", help="Input ONNX model in QDQ format (name under Checkpoints/ without .onnx extension)")
    parser.add_argument("output_onnx", help="Output ONNX model name in QOp format (name under Checkpoints/ without .onnx extension)")
    args = parser.parse_args()
    
    convert_qdq_to_qop(args.input_onnx, args.output_onnx)


if __name__ == "__main__":
    main()

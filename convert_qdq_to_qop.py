import argparse
import sys
import onnx
from onnx import helper, numpy_helper

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

def convert_qdq_to_qop(model_path, output_path):
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph

    new_nodes = []
    outputs_to_remove = set()
    
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
            
            # Get the quantized inputs (shared by both QLinearConv and ConvInteger)
            x_q_name = x_dq.input[0]
            x_scale = x_dq.input[1]
            x_zp = x_dq.input[2] if len(x_dq.input) > 2 else ""
            
            w_q_name = w_dq.input[0]
            w_scale = w_dq.input[1]
            w_zp = w_dq.input[2] if len(w_dq.input) > 2 else ""

            if len(consumers) == 1 and consumers[0].op_type == 'QuantizeLinear':
                # DequantizeLinear (x2) -> Conv -> QuantizeLinear  =>  QLinearConv
                y_q = consumers[0]
                y_scale = y_q.input[1]
                y_zp = y_q.input[2] if len(y_q.input) > 2 else ""
                q_conv_output = y_q.output[0]

                inputs = [
                    x_q_name, x_scale, x_zp,
                    w_q_name, w_scale, w_zp,
                    y_scale, y_zp
                ]
                if b_name:
                    inputs.append(b_name)  # bias must be int32 for QLinearConv

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
                # DequantizeLinear (x2) -> Conv  (no following QuantizeLinear)  =>  ConvInteger
                inputs = [x_q_name, w_q_name]
                if x_zp:
                    inputs.append(x_zp)
                if w_zp:
                    # x_zp slot must be filled if w_zp is provided
                    if not x_zp:
                        inputs.append("")
                    inputs.append(w_zp)

                new_node = helper.make_node(
                    'ConvInteger',
                    inputs=inputs,
                    outputs=[node.output[0]],
                    name=node.name + "_integer",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                print(f"Replaced {node.name} (Conv) with ConvInteger")

        elif node.op_type == 'MatMul':
            x_name = node.input[0]
            w_name = node.input[1]
            
            x_dq = get_node_by_output(graph.node, x_name)
            w_dq = get_node_by_output(graph.node, w_name)
            
            if not (x_dq and w_dq and x_dq.op_type == 'DequantizeLinear' and w_dq.op_type == 'DequantizeLinear'):
                continue

            y_name = node.output[0]
            consumers = find_consumers(graph.node, y_name)

            # Get the quantized inputs (shared by both QLinearMatMul and MatMulInteger)
            x_q_name = x_dq.input[0]
            x_scale = x_dq.input[1]
            x_zp = x_dq.input[2] if len(x_dq.input) > 2 else ""

            w_q_name = w_dq.input[0]
            w_scale = w_dq.input[1]
            w_zp = w_dq.input[2] if len(w_dq.input) > 2 else ""

            if len(consumers) == 1 and consumers[0].op_type == 'QuantizeLinear':
                # DequantizeLinear (x2) -> MatMul -> QuantizeLinear  =>  QLinearMatMul
                y_q = consumers[0]
                y_scale = y_q.input[1]
                y_zp = y_q.input[2] if len(y_q.input) > 2 else ""
                q_matmul_output = y_q.output[0]

                inputs = [
                    x_q_name, x_scale, x_zp,
                    w_q_name, w_scale, w_zp,
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
                # DequantizeLinear (x2) -> MatMul  (no following QuantizeLinear)  =>  MatMulInteger
                inputs = [x_q_name, w_q_name]
                if x_zp:
                    inputs.append(x_zp)
                if w_zp:
                    if not x_zp:
                        inputs.append("")  # pad x_zp slot
                    inputs.append(w_zp)

                new_node = helper.make_node(
                    'MatMulInteger',
                    inputs=inputs,
                    outputs=[node.output[0]],
                    name=node.name + "_integer",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                print(f"Replaced {node.name} (MatMul) with MatMulInteger")

    if not new_nodes:
        print("No QDQ patterns found to replace.")
        return
        
    final_nodes = []
    for node in graph.node:
        if node.output[0] not in outputs_to_remove:
            final_nodes.append(node)
            
    final_nodes.extend(new_nodes)
    
    # Topological sort
    ready_tensors = set([i.name for i in graph.input])
    ready_tensors.update([i.name for i in graph.initializer])
    
    unsorted_nodes = final_nodes.copy()
    sorted_nodes = []
    
    progress = True
    while unsorted_nodes and progress:
        progress = False
        remaining = []
        for node in unsorted_nodes:
            if all(inp in ready_tensors or inp == "" for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    ready_tensors.add(out)
                progress = True
            else:
                remaining.append(node)
        unsorted_nodes = remaining
        
    if unsorted_nodes:
        print("Warning: Graph may not be fully connected or has cycles!")
        sorted_nodes.extend(unsorted_nodes)
    
    # Create new graph
    new_graph = helper.make_graph(
        sorted_nodes,
        graph.name,
        graph.input,
        graph.output,
        graph.initializer,
        graph.value_info
    )
    
    new_model = helper.make_model(new_graph, producer_name='qdq-to-qop-converter', opset_imports=model.opset_import)
    
    print(f"Saving QOp model to {output_path}...")
    onnx.save(new_model, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX QDQ format to QOp format")
    parser.add_argument("input_onnx", help="Input ONNX model in QDQ format")
    parser.add_argument("output_onnx", help="Output ONNX model path in QOp format")
    args = parser.parse_args()
    
    convert_qdq_to_qop(args.input_onnx, args.output_onnx)

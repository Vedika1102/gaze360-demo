"""Export the L2CS-Net checkpoint to ONNX + INT8-quantized ONNX for fast CPU
inference, and verify numerical parity against the PyTorch model.

    python export_onnx.py --weights gaze360_model.pth.tar
"""
import argparse

import numpy as np
import torch

from l2cs_model import load_l2cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="gaze360_model.pth.tar")
    ap.add_argument("--fp32", default="models/l2cs.onnx")
    ap.add_argument("--int8", default="models/l2cs.int8.onnx")
    args = ap.parse_args()

    model = load_l2cs(args.weights, "cpu")
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model, dummy, args.fp32,
        input_names=["input"], output_names=["yaw", "pitch"],
        dynamic_axes={"input": {0: "n"}, "yaw": {0: "n"}, "pitch": {0: "n"}},
        opset_version=17, dynamo=False)
    print(f"exported {args.fp32}")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(args.fp32, args.int8, weight_type=QuantType.QInt8)
    print(f"quantized {args.int8}")

    # parity check
    import onnxruntime as ort
    with torch.no_grad():
        ty, tp = model(dummy)
    for path in (args.fp32, args.int8):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        oy, op = sess.run(None, {"input": dummy.numpy()})
        dy = float(np.abs(ty.numpy() - oy).max())
        dp = float(np.abs(tp.numpy() - op).max())
        print(f"{path}: max|dyaw|={dy:.4f} max|dpitch|={dp:.4f}")


if __name__ == "__main__":
    main()

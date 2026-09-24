import onnxruntime as ort
import numpy as np

MODEL_PATH = "/home/project/projecteye/object_onnx/yolov5n.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0].name

# dummy input in FP16
dummy = np.random.rand(1, 3, 640, 640).astype(np.float16)

out = session.run(None, {inp: dummy})

print("\n=== OUTPUT SHAPES ===")
for i, o in enumerate(out):
    arr = np.array(o)
    print(f"Output {i}: shape={arr.shape}, dtype={arr.dtype}")

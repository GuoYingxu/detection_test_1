import onnxruntime as ort
print("available:", ort.get_available_providers())
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] 
sess = ort.InferenceSession("unet.onnx",providers=providers)
print("session providers:", sess.get_providers())

print(ort.__version__)
print(ort.get_device())
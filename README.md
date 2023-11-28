# onnxruntime-web-on-extension
Sample code of chrome browser extension with onnxruntime-web running in service worker.

# Usage
```bash
# 1. prepare onnx model file
# NOTE: the sample onnx model file has been created at python_scripts/model.onnx. To reproduce the creation process, run `python3 python_scripts/create_onnx_model.py`.
#python3 python_scripts/create_onnx_model.py
cp python_scripts/model.onnx ext/src/public/

# 2. build extension
cd ext
npm install
npm run build
```

This builds the extension in `/ext/dist` directory.

## Test Environment
To build extension:
- Node.js: v20.10.0
- npm: 10.2.3

To prepare onnx model file:
- python3: 3.10.6
  - torch: 2.1.0

# License
MIT

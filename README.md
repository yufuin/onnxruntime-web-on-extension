# onnxruntime-web-on-extension
Sample code of chrome browser extension with onnxruntime-web running in service worker.

build tool: TypeScript + Vite

# Usage
```bash
# 0. download git repository
git clone https://github.com/yufuin/onnxruntime-web-on-extension.git
cd onnxruntime-web-on-extension

# 1. prepare onnx model file
# NOTE: the sample onnx model file has been created at python_scripts/model.onnx. To reproduce the creation, run `python3 python_scripts/create_onnx_model.py`.
#python3 python_scripts/create_onnx_model.py
cp python_scripts/model.onnx ext/src/public/

# 2. build extension
cd ext
npm install
npm run build
```

The built extension is located in the `REPOSITORY_ROOT/ext/dist` directory.
To test the extension, load the built directory as an unpacked extension on your browser and open the service_worker console.

## Test Environment
To build extension:
- Node.js: v20.10.0
- npm: 10.2.3

To prepare onnx model file:
- python3: 3.10.6
  - torch: 2.1.0

# License
MIT

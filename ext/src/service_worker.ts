import * as ort from "onnxruntime-web";
ort.env.wasm.numThreads = 1; // ref: https://github.com/microsoft/onnxruntime/issues/14445
ort.env.wasm.wasmPaths = "./";

async function test_ort() {
    try {
        // setup session
        const session = await ort.InferenceSession.create("./model.onnx", {executionProviders: ["wasm"]});
        console.log("session:", session);
        console.log(`input names: ${session.inputNames}`);
        console.log(`output names: ${session.outputNames}`);

        // prepare input
        const batch_size = 1;
        const input_dim = 2;
        const input_tensor_data = new Float32Array([2.5, 4.25]); // the data buffer is a flattened tensor (shape=[1,2] => num_elems=[1*2]=[2]).
        const input_tensor = new ort.Tensor("float32", input_tensor_data, [batch_size, input_dim]);
        console.log(`flattened input tensor: [${input_tensor.data}] (original shape=[${input_tensor.dims}])`);

        // forward
        // input and output names (here `x` and `y`) depend on the model definition.
        const feeds = { x: input_tensor };
        const results = await session.run(feeds);
        const output_tensor = results.y;
        console.log(`flattened output tensor: [${output_tensor.data}] (original shape=[${output_tensor.dims}])`);

        console.log("session successfully run.");

    } catch (error) {
        console.log("error:", error);
    }
}
test_ort();

console.log(`service_worker loaded.`);

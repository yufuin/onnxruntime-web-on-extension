async function test_ort_content_scripts() {
    const ort = await import("onnxruntime-web/webgpu");
    ort.env.wasm.wasmPaths = chrome.runtime.getURL("./");
    try {
        // setup session
        const session = await ort.InferenceSession.create(chrome.runtime.getURL("./model.onnx"), {executionProviders: ["webgpu"]});
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

        session.release();
        console.log("session successfully run.");

    } catch (error) {
        console.log("error:", error);
    }
}
test_ort_content_scripts();
console.log("hello from content_scripts.ts");

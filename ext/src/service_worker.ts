import * as ort from 'onnxruntime-web';
ort.env.wasm.numThreads = 1; // ref: https://github.com/microsoft/onnxruntime/issues/14445
// ort.env.wasm.wasmPaths = './';

async function test_ort() {
    try {
        // setup session
        const session = await ort.InferenceSession.create('./model.onnx');
        console.log('session:', session);
        console.log(`input names: ${session.inputNames}`);
        console.log(`output names: ${session.outputNames}`);

        // prepare input
        const input_dim = 2;
        const input_tensor_data = new Float32Array([2.0,4.0, 5.0,-2.0, 7.0,3.0]); // tensor data is flattened array. (batch_size=3 * dim=2 -> num_elems=6)
        const batch_size = input_tensor_data.length / input_dim;
        const input_tensor = new ort.Tensor('float32', input_tensor_data, [batch_size, input_dim]);

        // forward
        // input and output names (here `x` and `y`) depend on the model definition.
        const feeds = { x: input_tensor };
        const results = await session.run(feeds);
        const output_tensor = results.y;
        console.log(`input (flattened): ${input_tensor.data} (shape=[${input_tensor.dims}])`);
        console.log(`output (flattened): ${output_tensor.data} (shape=[${output_tensor.dims}])`);

        // release resources
        session.release();
        console.log('session successfully run.');

    } catch (error) {
        console.log('error:', error);
    }
}
test_ort();

console.log(`service_worker loaded.`);

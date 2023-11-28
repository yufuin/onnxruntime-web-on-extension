# %%
"""
this creates sample onnx model file.
to use the created model, put it in `src/public/` of the extension directory.
"""

import pathlib
SAVE_PATH = pathlib.Path(__file__).resolve().parent / "model.onnx"

import torch

# %% prepare model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2,3)
    def forward(self, x):
        h = self.fc1(x)
        return h
    
model = Model()

model.fc1.bias.data = torch.zeros([3])
model.fc1.weight.data = torch.FloatTensor([
    [1.0, 0.0], # first value
    [0.0, 1.0], # second value
    [1.0, 1.0], # sum
])

# %% save
sample_input = {"x":torch.FloatTensor([[3.0, 2.0]])}

torch.onnx.export(
    model,
    sample_input,
    SAVE_PATH,
    input_names=["x"],
    output_names=["y"],
    dynamic_axes={
        "x": {0:"batch_size"},
    },
)
print(f"saved onnx model file. (path={SAVE_PATH})")
print('to use the created model, put it in `src/public/` of the extension directory.')

# %%

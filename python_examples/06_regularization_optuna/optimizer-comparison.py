import torch

def make_optimizers(parameters):
    return {
        "SGD": torch.optim.SGD(parameters, lr=0.05),
        "SGD+momentum": torch.optim.SGD(parameters, lr=0.05, momentum=0.9),
        "Adam": torch.optim.Adam(parameters, lr=0.001),
        "AdamW": torch.optim.AdamW(parameters, lr=0.001, weight_decay=1e-4),
    }

parameter = torch.nn.Parameter(torch.ones(2))
optimizers = make_optimizers([parameter])
assert set(optimizers) == {"SGD", "SGD+momentum", "Adam", "AdamW"}
assert optimizers["SGD+momentum"].param_groups[0]["momentum"] == 0.9
assert optimizers["AdamW"].param_groups[0]["weight_decay"] == 1e-4
print("SGD, momentum, Adam and AdamW are configured")

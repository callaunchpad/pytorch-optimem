import torch
import torch.nn as nn

cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda:0')

READ_DEVICE = gpu_device

usage_history = []


def reset_usage_history():
    global usage_history
    temp = usage_history
    usage_history = []
    return temp


def read_gpu_memory():
    return torch.cuda.memory_allocated(READ_DEVICE)


def store_memory():
    usage_history.append(read_gpu_memory())


class ReadLayer(nn.Module):
    def forward(self, x):
        store_memory()
        return x


def add_read(layer):
    test = [layer] + [ReadLayer()]
    return nn.Sequential(*test)


def layer_size(layer):
    return sum(e.flatten().shape[0] for e in layer.parameters())


def get_base_children(model, max_layer_size=-1):
    base_children = []

    # Get all base layers (with no children)
    def get_children(module_par):
        modules = list(module_par.named_modules())
        for name, module in modules[1:]:
            if '.' in name:
                continue
            if (len(list(module.children())) == 0 or layer_size(module) < max_layer_size) and (type(module) != nn.ModuleList):
                base_children.append((module_par, name, module))
            else:
                get_children(module)

    get_children(model)
    return base_children


def add_reads(model):
    base_children = get_base_children(model)
    for par, name, module in base_children:
        setattr(par, name, add_read(module))

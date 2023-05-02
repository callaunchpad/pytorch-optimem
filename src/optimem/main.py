import torch
import torch.nn as nn

CPU_DEVICE = torch.device('cpu')

class FlexSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class LayerToDevice(nn.Module):
    def __init__(self, device, layer):
        super().__init__()
        self.D = device
        self.layer = [layer]
    def forward(self, *args):
        self.layer[0].to(self.D)
        if(len(args) == 1):
            return args[0]
        return args
    
class LayerOffDevice(nn.Module):
    def __init__(self, device, layer):
        super().__init__()
        self.D = device
        self.layer = [layer]
        
    def forward(self, *args):
        self.layer[0] = self.layer[0].to(CPU_DEVICE)
        if(len(args) == 1):
            return args[0]
        return args
    
class DataToDevice(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, *args):
        ret = tuple([e.to(self.device) if type(e) == torch.Tensor else e for e in args])
        if(len(ret) == 1):
            return ret[0]
        return ret
    
class OptimemPageLayer(FlexSequential):
    def __init__(self, layer, device):
        layers = [LayerToDevice(device, layer)] + [layer] + [LayerOffDevice(device, layer)]
        super().__init__(*layers)
        
class OptimemChunkLayer(FlexSequential):
    def __init__(self, layer, device):
        layers = [DataToDevice(device)] + [layer] 
        super().__init__(*layers)
        
def _layer_size(layer):
    return sum(e.flatten().shape[0] for e in layer.parameters())

def _get_base_children(model, max_layer_size=-1):
    ''' Get sub-modules which are < max_layer_size or have no children'''
    base_children = []

    def get_children(module_par):
        modules = list(module_par.named_modules())
        for name, module in modules[1:]:
            if '.' in name:
                continue
            if (len(list(module.children())) == 0 or _layer_size(module) < max_layer_size) and (type(module) != nn.ModuleList):
                base_children.append((module_par, name, module))
            else:
                get_children(module)

    get_children(model)
    return base_children

def page(model, device = torch.device('cuda'), max_layer_size = -1):
    assert not model.training, "Model should be in evaluation mode"
    assert all([p.device.type == 'cpu' for p in model.parameters()]), "Model should be on CPU"
    
    base_children = _get_base_children(model, max_layer_size)

    for par, name, module in base_children:
        setattr(par, name, OptimemPageLayer(module, device))
        
def chunk(model, device = torch.device('cuda'), max_capacity = 1e9):
    
    assert not model.training, "Model should be in evaluation mode"
    assert all([p.device.type == 'cpu' for p in model.parameters()]), "Model should be on CPU"
    
    base_children = _get_base_children(model, max_capacity)
    stored = 0
    
    for par, name, module in base_children:
        stored += _layer_size(module)
        if stored > max_capacity:
            setattr(par, name, OptimemChunkLayer(module, CPU_DEVICE))
        else:
            module.to(device)
            setattr(par, name, OptimemChunkLayer(module, device))


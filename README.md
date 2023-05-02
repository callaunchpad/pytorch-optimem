# pytorch-optimem
A Python package for reducing memory footprint of PyTorch models

It is well known that GPUs are much faster for deep learning applications as they are highly parallelizable. The goal of PyTorch Optimem is to enable larger models to take advantage of a GPU even if they cannot contain the entire model in VRAM. This is accomplished by either paging memory to and from the GPU or maximize usage of the GPU. Many of the design considerations for Optimem have a focus on model inference, if you are using this repository for training is it likely that something will break.

### Usage

Install the package using pip
`pip install --upgrade pytorch-optimem`

PyTorch Optimem offers two modes, `page` or `chunk`. For both modes, the model must be on the GPU.

#### Paging

Optimem in paging mode attempts to GPU page the input model for all layers. This means we move pieces of the model to and from the GPU which has it's own pros and cons. 

NOTE: The data tensor must start on GPU.

```python
import optimem

resnet = torchvision.models.resnet101(pretrained=True).eval()
paged_resent = optimem.page(resnet)
```

There are two optional parameters, device which is the device to page the model into (typically a GPU), and max_layer_size which is the maximum number of parameters we wish to page at a time between the CPU and GPU.

#### Chunking

Optimem in chunking mode attempts to identify the largest chunk of the model which can be loaded in the GPU and stores that in the GPU VRAM. 

NOTE: The data tensor must start on CPU.

```python
import optimem

resnet = torchvision.models.resnet101(pretrained=True).eval()
chunked_resnet = optimem.chunk(resnet)
```

There are two optional parameters, device which is the device we will chunk the model into (typically a GPU), and max_capacity which is the maximum number of parameters we wish to store on the target device.



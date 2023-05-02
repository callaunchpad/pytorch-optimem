# pytorch-optimem
A Python package for reducing memory footprint of PyTorch models

It is well known that GPUs are much faster for deep learning applications as they are highly parallelizable. The goal of PyTorch Optimem is to enable larger models to take advantage of a GPU even if they cannot contain the entire model in VRAM. This is accomplished by either paging memory to and from the GPU or maximize usage of the GPU. Many of the design considerations for Optimem have a focus on model inference, if you are using this repository for training is it likely that something will break.

### Usage

Install the package using pip
`pip install --upgrade pytorch-optimem`

PyTorch Optimem offers two modes, `page` or `chunk`. 

#### Paging

Optimem in paging mode attempts to GPU page the input model for all layers. This means we move pieces of the model to and from the GPU which has it's own pros and cons.

```python
from optimem import page

resnet = torchvision.models.resnet101(pretrained=True).eval()
paged_resent = page(resnet)
```

#### Chunking

Optimem in chunking mode attempts to identify the largest chunk of the model which can be loaded in the GPU and stores that in the GPU VRAM. 

```python
from optimem import chunk

resnet = torchvision.models.resnet101(pretrained=True).eval()
chunked_resnet = chunk(resnet)
```

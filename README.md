# PyTorch Optimem

A package for speeding up inference of PyTorch modules with low-memory GPUs.

GPUs can make your PyTorch deep learning applications run significantly faster, but as models get larger, it's harder and harder to fit them within this specialized hardware. Traditionally, if your model is larger than the GPUs memory (vRAM), you have to run the entirety on the much slower CPU. The goal of PyTorch Optimem is to take advantage of GPUs of any size, even if they cannot contain the entire model through **paging** or **chunk loading**. 

This currently focuses only on inference, but future versions may include similar techniques for training applications. Additionally, PyTorch Optimem supports paging and chunk loading on other devices (TPU, MPS, etc.) but has not benchmarked for those.

## Usage

Install the package using pip:

```
pip install --upgrade pytorch-optimem
```


PyTorch Optimem offers two modes, `page` or `chunk`. For both modes, the model must be in evaluation mode (run `model.eval()`) and all parameters must currently be on the CPU.

### Paging

Paging mode pages the input model between the RAM and vRAM during inference. At a customizable granularity, it moves chunks of the model to the GPU or other hardware "on the fly" while maintaining the data on the GPU throughout. An illustration of the process is shown below.

![Screen-Recording-2023-05-02-at-5 28 39-PM_1](https://user-images.githubusercontent.com/8518898/235813251-08200476-a4fb-4ec4-b3c0-b50cdca371e8.gif)

**Usage:**
```python
optimem.page(
  model: torch.nn.Module # model to apply paging to
  device: torch.device = torch.device('cuda') # deivce to page from CPU to
  max_layer_size: int = -1 # maximum number of parameters for which to stop recursing when determining granularity of paging
) - > None
```
**Example:**

```python
import torchvision
import optimem

resnet = torchvision.models.resnet101(pretrained=True).eval()
paged_resent = optimem.page(resnet)
```

NOTE: The data tensor must start on GPU.

NOTE: `max_layer_size` exists to customize how much GPU is being used. For example, ResNet has "block" modules with multiple conv layers + max pooling and if `max_layer_size` is set high enough those will be paged all at once instead of each layer being paged invidually.

Performance results for passing in 512x512 images into ResNet-101 are shown below:

<img width="672" alt="image" src="https://user-images.githubusercontent.com/8518898/235816037-832b0ae0-05a2-4c9b-a0c1-b2b42d1629e8.png">


### Chunking

Chunking mode identifies the largest chunk of the model which can be loaded to the GPU and permanently stores it in the vRAM. It also adds the pipelining that allows the model to convert the data to the right type throughout the process.

<img width="678" alt="image" src="https://user-images.githubusercontent.com/8518898/235816355-c28596e7-79f3-4ed3-95c8-3de66ae93e24.png">

*The first few layers are added to the GPU so that part of the model can run faster*

**Usage:**
```python
optimem.chunk(
  model: torch.nn.Module # model to apply paging to
  device: torch.device = torch.device('cuda') # deivce to page from CPU to
  max_capacity: int = 1e9 # total capacity of parameters that can be placed onto the GPU
) - > None
```
**Example:**

```python
import torchvision
import optimem

resnet = torchvision.models.resnet101(pretrained=True).eval()
paged_resent = optimem.chunk(resnet)
```

NOTE: The data tensor must start on CPU.
NOTE: The higher the max_capacity 

Performance results for passing in 512x512 images into ResNet-101 are shown below:
<img width="748" alt="image" src="https://user-images.githubusercontent.com/8518898/235816226-efd4d076-b6b8-47ef-ad9a-f71801b53833.png">




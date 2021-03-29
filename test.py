# conda install -c fastai -c pytorch -c anaconda fastai gh anaconda


# In[]:
import os
from pathlib import Path

import torch

from fastai.data.block import DataBlock, CategoryBlock, get_image_files, GrandparentSplitter, parent_label
from fastai.metrics import error_rate
from fastai.vision.augment import Resize, RandomResizedCrop, aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.all import cnn_learner, ClassificationInterpretation, load_learner, nn, partial, MixUp, xresnet50, \
    accuracy, top_k_accuracy, Learner
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from fastai.distributed import *
# In[]:
products = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name="train", valid_name="validation"),
    get_y=parent_label,
    item_tfms=Resize(192))

products = products.new(
    item_tfms=RandomResizedCrop(168, min_scale=0.8),
    batch_tfms=aug_transforms())

project_path = Path("/home/yaro/Workspace/fastai/")
dataset_path = project_path.joinpath("for_test")
dls = products.dataloaders(dataset_path)

gpu = None
if torch.cuda.is_available():
    if gpu is not None: torch.cuda.set_device(gpu)
    n_gpu = torch.cuda.device_count()
else:
    n_gpu = None

learn = cnn_learner(dls, resnet18, metrics=error_rate).to_fp16()

# The context manager way of dp/ddp, both can handle single GPU base case.
if gpu is None and n_gpu is not None:
    ctx = learn.parallel_ctx
    with partial(ctx, gpu)():
        print(f"Training in {ctx.__name__} context on GPU {list(range(n_gpu))}")
        learn.fine_tune(2)
else:
    learn.fine_tune(2)
# In[]:
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
plt.show()

# In[]:
learn.export()

# In[]: for test
#learn_inf = load_learner('export.pkl')
#learn_inf.predict(dataset_path.joinpath('validation/ananas/cam0.hand0.11054.jpg'))
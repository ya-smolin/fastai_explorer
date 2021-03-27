# conda install -c fastai -c pytorch -c anaconda fastai gh anaconda


# In[]:
import os

from fastai.data.block import DataBlock, CategoryBlock, get_image_files, GrandparentSplitter, parent_label
from fastai.metrics import error_rate
from fastai.vision.augment import Resize, RandomResizedCrop, aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.all import cnn_learner, ClassificationInterpretation, load_learner
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# In[]:
products = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name="train", valid_name="validation"),
    get_y=parent_label,
    item_tfms=Resize(192))

products = products.new(
    item_tfms=RandomResizedCrop(128, min_scale=0.5),
    batch_tfms=aug_transforms())

path = "/home/yaro/PycharmProjects/fastai_explorer/for_test"
dls = products.dataloaders(path)

learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

# In[]:
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
plt.show()

# In[]:
learn.export()

# In[]: for test
learn_inf = load_learner('export.pkl')
learn_inf.predict('for_test/validation/ananas/cam0.hand0.11054.jpg')
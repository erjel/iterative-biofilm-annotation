# Modified from 
# https://github.com/CSBDeep/CSBDeep/blob/master/examples/other/technical.ipynb

"""
BSD 3-Clause License

Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from csbdeep.internals.nets import common_unet

import numpy as np

from csbdeep.utils.tf import keras_import

keras = keras_import()

from skimage.segmentation import find_boundaries
from csbdeep.models import BaseModel, BaseConfig

from csbdeep.data import PadAndCropResizer
from csbdeep.utils import axes_check_and_normalize
from csbdeep.utils.tf import CARETensorBoardImage


import math

def crop(u,shape=(48, 96, 96)):
    """Crop central region of given shape"""
    return u[tuple(slice((s-m)//2,(s-m)//2+m) for s,m in zip(u.shape,shape))]

def to_3class_label(lbl, onehot=True):
    """Convert instance labeling to background/inner/outer mask"""
    b = find_boundaries(lbl,mode='outer')
    res = (lbl>0).astype(np.uint8)
    res[b] = 2
    if onehot:
        res = keras.utils.to_categorical(res,num_classes=3).reshape(lbl.shape+(3,))
    return res

def dice_bce_loss(n_labels):
    """Combined crossentropy and dice loss"""
    K = keras.backend
    def _sum(a):
        return K.sum(a, axis=(1,2), keepdims=True)
    def dice_coef(y_true, y_pred):
        return (2 * _sum(y_true * y_pred) + K.epsilon()) / (_sum(y_true) + _sum(y_pred) + K.epsilon())
    def _loss(y_true, y_pred):
        dice_loss = 0
        for i in range(n_labels):
            dice_loss += 1-dice_coef(y_true[...,i], y_pred[...,i])
        return dice_loss/n_labels + K.categorical_crossentropy(y_true, y_pred)
    return _loss

def datagen(X,Y,batch_size,seed=0):
    """Simple data augmentation"""
    g = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                                     rotation_range=10, shear_range=10, fill_mode='reflect')
    assert seed is not None
    gX = g.flow(X, batch_size=batch_size, seed=seed)
    gY = g.flow(Y, batch_size=batch_size, seed=seed)
    while True:
        yield gX.next(), gY.next()

def random_fliprot(img, mask, axis=None): 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x,y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y


class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size : (idx + 1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size : (idx + 1)*self.batch_size]
        
        samples = [augmenter(*sample) for sample in list(zip(batch_x, batch_y))]
        batch_x, batch_y = list(zip(*samples))
        
        return np.array(batch_x), np.array(batch_y)    

class SegConfig(BaseConfig):
    def __init__(self, unet_depth, **kwargs):
        super().__init__(**kwargs)
        self.unet_depth = unet_depth
        self.axes = 'ZYXC'


class SegModel(BaseModel):    
    @property
    def _config_class(self):
        return SegConfig
    
    def _build(self):
        return common_unet(n_dim=3, n_depth=self.config.unet_depth,
                           n_first=32, residual=False,
                           n_channel_out=self.config.n_channel_out,
                           last_activation='softmax')((None,None,None,self.config.n_channel_in))

    def _prepare_for_training(self, validation_data, lr):
        assert self.config.n_channel_out > 1        
        self.keras_model.compile(optimizer=keras.optimizers.Adam(lr),
                                 loss=dice_bce_loss(self.config.n_channel_out),
                                 metrics=['categorical_crossentropy','accuracy'])
        self.callbacks = self._checkpoint_callbacks()
        self.callbacks.append(keras.callbacks.TensorBoard(log_dir=str(self.logdir/'logs'),
                                                          write_graph=False, profile_batch=0))
        self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
                                                   log_dir=str(self.logdir/'logs'/'images'),
                                                   n_images=3, prob_out=False))
        self._model_prepared = True
        
    def train(self, X,Y, validation_data, lr, batch_size, epochs, steps_per_epoch):
        if not self._model_prepared:
            self._prepare_for_training(validation_data, lr)
            
        training_data = CustomDataGenerator(X,Y,batch_size)
        
        history = self.keras_model.fit(training_data, validation_data=validation_data,
                                       epochs=epochs, steps_per_epoch=steps_per_epoch,
                                       callbacks=self.callbacks, verbose=1)
        self._training_finished()
        return history
    
    def predict(self, img, axes=None, normalizer=None, resizer=PadAndCropResizer()):
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        axes_net = self.config.axes
        if axes is None:
            axes = axes_net
        axes = axes_check_and_normalize(axes, img.ndim)
        axes_net_div_by = tuple((2**self.config.unet_depth if a in 'XYZ' else 1) for a in axes_net)
        x = self._make_permute_axes(axes, axes_net)(img)
        x = normalizer(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)        
        pred = self.keras_model.predict(x[np.newaxis])[0]
        pred = resizer.after(pred, axes_net)
        return pred
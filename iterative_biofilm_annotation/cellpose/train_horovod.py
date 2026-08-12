import mxnet as mx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread, imsave
from skimage.segmentation import relabel_sequential
import argparse
import os

import os, sys, time, shutil, tempfile, datetime, pathlib, gc
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2

from mxnet import gluon, nd
import mxnet as mx

from cellpose import transforms, dynamics, utils, resnet_style, plot

import horovod.mxnet as hvd
from mxnet.io import DataBatch, DataIter
import argparse




class CellposeModel():
    """
    Parameters
    -------------------
    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available
    pretrained_model: str or list of strings (optional, default False)
        path to pretrained cellpose model(s), if False, no model loaded;
        if None, built-in 'cyto' model loaded
    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False
    batch_size: int (optional, default 8)
        number of 224x224 patches to run simultaneously on the GPU
        (can make smaller or bigger depending on GPU memory usage)
    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model
    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))
    """

    def __init__(self, gpu=True, pretrained_model=False, batch_size=8,
                    diam_mean=27., net_avg=True, device=mx.gpu(), unet=False):
        super(CellposeModel, self).__init__()
        
        
        if device is not None:
            self.device = device
        elif gpu and utils.use_gpu():
            self.device = mx.gpu()
            print('>>>> using GPU')
        else:
            self.device = mx.cpu()
            print('>>>> using CPU')

        self.unet = unet
        if unet:
            nout = 1
        else:
            nout = 3

        self.pretrained_model = pretrained_model
        self.batch_size=batch_size
        self.diam_mean = diam_mean

        nbase = [32,64,128,256]
        self.net = resnet_style.CPnet(nbase, nout=nout)
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.initialize(ctx = self.device)#, grad_req='null')

        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')

        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_parameters(pretrained_model)
        elif pretrained_model is None and not unet:
            if net_avg:
                pretrained_model = [os.fspath(model_dir.joinpath('cyto_%d'%j)) for j in range(4)]
                if not os.path.isfile(pretrained_model[0]):
                    download_model_weights()
            else:
                pretrained_model = os.fspath(model_dir.joinpath('cyto_0'))
                if not os.path.isfile(pretrained_model):
                    download_model_weights()
                self.net.load_parameters(pretrained_model)
            self.diam_mean = 27.
            self.pretrained_model = pretrained_model

    def eval(self, x, channels=None, invert=False, rescale=None, do_3D=False, net_avg=True, 
             tile=True, flow_threshold=0.4, cellprob_threshold=0.0, compute_masks=True, progress=None):
        """
            segment list of images x, or 4D array - Z x nchan x Y x X
            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].
            invert: bool (optional, default False)
                invert image pixel intensity before running network
            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0
            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input
            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False
            tile: bool (optional, default True)
                tiles image for test time augmentation and to ensure GPU memory usage limited (recommended)
            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)
            cellprob_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)
            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.
            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI
            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels
            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0
            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image
        """
        nimg = len(x)
        if channels is not None:
            if len(channels)==2:
                if not isinstance(channels[0], list):
                    channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i], invert=invert) for i in range(nimg)]
        elif do_3D:
            for i in range(len(x)):
                if x[i].ndim<3:
                    raise ValueError('ERROR: cannot process 2D images in 3D mode') 
                elif x[i].ndim<4:
                    x[i] = x[i][...,np.newaxis]
                if x[i].shape[1]<4:
                    x[i] = np.transpose(x[i], (1,0,2,3))
                elif x[i].shape[-1]<4:
                    x[i] = np.transpose(x[i], (3,0,1,2))
                # put channels first
                if x[i].shape[0]>2:
                    print('WARNING: more than 2 channels given, use "channels" input for specifying channels - just using first two channels to run processing')
                    x[i] = x[i][:2]
            
        styles = []
        flows = []
        masks = []
        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        if nimg > 1:
            iterator = trange(nimg)
        else:
            iterator = range(nimg)

        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_parameters(self.pretrained_model[0])
            self.net.collect_params().grad_req = 'null'

        if not do_3D:
            for i in iterator:
                img = x[i].copy()
                if img.shape[0]<3:
                    img = np.transpose(img, (1,2,0))
                Ly,Lx = img.shape[:2]
                if img.shape[-1]==1:
                    img = np.concatenate((img, 0.*img), axis=-1)
                #tic=time.time()
                if isinstance(self.pretrained_model, str) or not net_avg:
                    y, style = self._run_net(img, rescale[i], tile)
                else:
                    y, style = self._run_many(img, rescale[i], tile)
                if progress is not None:
                    progress.setValue(55)
                styles.append(style)
                if compute_masks:
                    cellprob = y[...,-1]
                    if not self.unet:
                        dP = np.stack((y[...,0], y[...,1]), axis=0)
                        niter = 1 / rescale[i] * 200
                        p = dynamics.follow_flows(-1 * dP  / 5. , niter=niter)
                        if progress is not None:
                            progress.setValue(65)
                        maski = dynamics.get_masks(p, iscell=(cellprob>cellprob_threshold),
                                                   flows=dP, threshold=flow_threshold)
                        if progress is not None:
                            progress.setValue(75)
                        dZ = np.zeros((1,Ly,Lx), np.uint8)
                        dP = np.concatenate((dP, dZ), axis=0)
                        flow = plot.dx_to_circ(dP)
                        flows.append([flow, dP, cellprob, p])
                        maski = dynamics.fill_holes(maski)
                        masks.append(maski)
                else:
                    flows.append([None]*3)
                    masks.append([])
        else:
            for i in iterator:
                sstr = ['XY', 'XZ', 'YZ']
                if x[i].shape[-1] < 3:
                    x[i] = np.transpose(x[i], (3,0,1,2))
                pm = [(1,2,3,0), (2,1,3,0), (3,1,2,0)]
                ipm = [(0,1,2,3), (0,2,1,3), (0,2,3,1)]
                tic=time.time()
                flowi=[]
                for p in range(3):
                    xsl = np.transpose(x[i].copy(), pm[p])
                    flowi.append(np.zeros(((3,xsl.shape[0],xsl.shape[1],xsl.shape[2])), np.float32))
                    # per image
                    ziterator = trange(xsl.shape[0])
                    print('\n running %s (%d, %d) \n\n'%(sstr[p], xsl.shape[1], xsl.shape[2]))
                    for z in ziterator:
                        if isinstance(self.pretrained_model, str) or not net_avg:
                            y, style = self._run_net(xsl[z], rescale[0], tile=tile)
                        else:
                            y, style = self._run_many(xsl[z], rescale[0], tile=tile)
                        y = np.transpose(y[:,:,[1,0,2]], (2,0,1))
                        flowi[p][:,z] = y
                    flowi[p] = np.transpose(flowi[p], ipm[p])
                    if progress is not None:
                        progress.setValue(25+15*p)
                dX = flowi[0][0] + flowi[1][0]
                dY = flowi[0][1] + flowi[2][0]
                dZ = flowi[1][1] + flowi[2][1]
                cellprob = flowi[0][-1] + flowi[1][-1] + flowi[2][-1]
                dP = np.concatenate((dZ[np.newaxis,...], dY[np.newaxis,...], dX[np.newaxis,...]), axis=0)
                print('flows computed %2.2fs'%(time.time()-tic))
                yout = dynamics.follow_flows(-1 * dP / 5.)
                print('dynamics computed %2.2fs'%(time.time()-tic))
                maski = dynamics.get_masks(yout, iscell=(cellprob>cellprob_threshold))
                print('masks computed %2.2fs'%(time.time()-tic))
                flow = np.array([plot.dx_to_circ(dP[1:,i]) for i in range(dP.shape[1])])
                flows.append([flow, dP, cellprob, yout])
                masks.append(maski)
                styles.append([])
        return masks, flows, styles

    def _run_many(self, img, rsz=1.0, tile=True):
        """ loop over netwroks in pretrained_model and average results
        Parameters
        --------------
        img: float, [Ly x Lx x nchan]
        rsz: float (optional, default 1.0)
            resize coefficient for image
        tile: bool (optional, default True)
            tiles image for test time augmentation and to ensure GPU memory usage limited (recommended)
        Returns
        ------------------
        yup: array [3 x Ly x Lx]
            yup is output averaged over networks;
            yup[0] is Y flow; yup[1] is X flow; yup[2] is cell probability
        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles,
            but not averaged over networks.
        """
        for j in range(len(self.pretrained_model)):
            self.net.load_parameters(self.pretrained_model[j])
            self.net.collect_params().grad_req = 'null'
            yup0, style = self._run_net(img, rsz, tile)
            if j==0:
                yup = yup0
            else:
                yup += yup0
        yup = yup / len(self.pretrained_model)
        return yup, style

    def _run_tiled(self, imgi, bsize=224):
        """ run network in tiles of size [bsize x bsize]
        First image is split into overlapping tiles of size [bsize x bsize].
        Then 4 versions of each tile are created:
            * original
            * flipped vertically
            * flipped horizontally
            * flipped vertically and horizontally
        The average of the network output over tiles is returned.
        Parameters
        --------------
        imgi: array [nchan x Ly x Lx]
        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
        Returns
        ------------------
        yf: array [3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability
        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles
        """
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize, augment=True)
        nbatch = self.batch_size
        niter = int(np.ceil(IMG.shape[0]/nbatch))
        y = np.zeros((IMG.shape[0], 3, bsize, bsize))
        for k in range(niter):
            irange = np.arange(nbatch*k, min(IMG.shape[0], nbatch*k+nbatch))
            img = nd.array(IMG[irange], ctx=self.device)
            y0, style = self.net(img)
            y[irange] = y0.asnumpy()
            if k==0:
                styles = style.asnumpy()[0]
            styles += style.asnumpy().sum(axis=0)
        styles /= IMG.shape[0]
        y = transforms.unaugment_tiles(y)
        yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
        styles /= (styles**2).sum()**0.5
        del IMG
        gc.collect()
        return yf, styles

    def _run_net(self, img, rsz=1.0, tile=True, bsize=224):
        """ run network on image
        Parameters
        --------------
        img: array [Ly x Lx x nchan]
        rsz: float (optional, default 1.0)
            resize coefficient for image
        tile: bool (optional, default True)
            tiles image for test time augmentation and to ensure GPU memory usage limited (recommended)
        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
        Returns
        ------------------
        y: array [3 x Ly x Lx]
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability
        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles
        """
        shape = img.shape
        if abs(rsz - 1.0) < 0.03:
            rsz = 1.0
            Ly,Lx = img.shape[:2]
        else:
            Ly = int(img.shape[0] * rsz)
            Lx = int(img.shape[1] * rsz)
            img = cv2.resize(img, (Lx, Ly))

        # make image nchan x Ly x Lx for net
        if img.ndim<3:
            img = np.expand_dims(img, axis=-1)
        img = np.transpose(img, (2,0,1))

        # pad for net so divisible by 4
        img, ysub, xsub = transforms.pad_image_ND(img)
        if tile:
            y,style = self._run_tiled(img, bsize)
            y = np.transpose(y[:3], (1,2,0))
        else:
            img = nd.array(np.expand_dims(img, axis=0), ctx=self.device)
            y,style = self.net(img)
            img = img.asnumpy()
            y = np.transpose(y[0].asnumpy(), (1,2,0))
            style = style.asnumpy()[0]
            style = np.ones(10)

        y = y[np.ix_(ysub, xsub, np.arange(3))]
        style /= (style**2).sum()**0.5
        if rsz!=1.0:
            y = cv2.resize(y, (shape[1], shape[0]))
        return y, style

    def train(self, train_data, val_data, params=None, channels=[0, 0],
              pretrained_model=None, save_path=None, save_every=10,
              learning_rate=0.2, n_epochs=10, weight_decay=0.00001, batch_size=8, rescale=True):
        

        d = datetime.datetime.now()
        self.learning_rate = learning_rate*hvd.size()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = 0.9
        
        run_test = True
        
        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            print('WARNING: no save_path given, model not saving')
        
        # compute average cell diameter
        if rescale:
            scale_range = 0.5
        else:
            scale_range = 1.0
            
        nimg = train_data.patch_num

        print('>>>> training network with %d channel input <<<<'%len(channels))
        print('>>>> saving every %d epochs'%save_every)
        print('>>>> median diameter = %d'%self.diam_mean)
        print('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate, self.batch_size, self.weight_decay))
        print('>>>> ntrain = %d'%nimg)
        if run_test:
            print('>>>> ntest = %d'%val_data.patch_num)

        criterion  = gluon.loss.L2Loss()
        criterion2 = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        
        opt_params = {'learning_rate': self.learning_rate, 'momentum': self.momentum, 'wd': self.weight_decay}
        opt = mx.optimizer.create('sgd', **opt_params)
        
        if params is None:
            print('Please define params for horovod training!')
            return
        
        trainer = hvd.DistributedTrainer(params, opt)

        
        warmup_lr = np.linspace(0, self.learning_rate, 10)
        tic = time.time()

        
        def forward(data, flows, diams, train=True, scale_range=.0):
            if rescale:
                rsc = np.asarray(diams) / self.diam_mean
            else:
                rsc = np.ones(batch_size, np.float32)

            # augmentation    
            imgi, lbl, _ = transforms.random_rotate_and_resize(data, Y=flows,
                                        rescale=rsc, scale_range=scale_range)
                
            X = nd.array(imgi, ctx=self.device)
                
            if not self.unet:
                veci = 5. * nd.array(lbl[:,1:], ctx=self.device)
                    
            lbl  = nd.array(lbl[:,0]>.5, ctx=self.device)
                
            if train:
                with mx.autograd.record():
                    y, style = self.net(X)
                    if self.unet:
                        loss = criterion2(y[:,-1] , lbl)
                    else:
                        loss = criterion(y[:,:-1] , veci) + criterion2(y[:,-1] , lbl)

                loss.backward()  
            else:
                y, style = self.net(X)
                if self.unet:
                    loss = criterion2(y[:,-1] , lbl)
                else:
                    loss = criterion(y[:,:-1] , veci) + criterion2(y[:,-1] , lbl)

            train_loss = nd.sum(loss).asscalar()
            return train_loss, len(loss)
        
        
        ksave = 0
        for iepoch in range(self.n_epochs):
            train_data.reset()
            lavg, nsum = 0, 0
            
            # learning rate warmup            
            if iepoch<len(warmup_lr):
                LR = warmup_lr[iepoch]
                trainer.set_learning_rate(LR)
                
            for ibatch, (data, flows, diams) in enumerate(train_data):
                
                
                train_loss, len_loss = forward(list(data), list(flows), diams, train=True, scale_range=scale_range)
                
                lavg += train_loss
                nsum += len_loss
                
                
                if iepoch>0:
                    trainer.step(batch_size)
                    
            if iepoch>self.n_epochs-100 and iepoch%10==1:
                LR = LR/2
                trainer.set_learning_rate(LR)

            if (iepoch%10==0 or iepoch<10) and hvd.rank() ==0:
                lavg = lavg / nsum
                
                if run_test:
                    val_data.reset()
                    lavgt = 0
                    ntsum = 0
                    
                    for ibatch, (data, flows, diams) in enumerate(val_data):
                        train_loss, len_loss = forward(list(data), list(flows), diams, train=False, scale_range=.0)
                        lavgt += train_loss
                        ntsum += len_loss
                        
                    (iepoch == 0 and ibatch == 0) and print('data shape:', data[0].shape, '\nlen_loss:', len_loss, flush=True)
                        
                print('Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f'%
                    (iepoch, time.time()-tic, lavg, lavgt/ntsum, LR))
                    
                    
            if save_path is not None and hvd.rank() == 0:
                if iepoch==self.n_epochs-1 or iepoch%save_every==1:
                    # save model at the end
                    file = 'cellpose_{}_{}_{}'.format(self.unet, file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                    ksave += 1
                    print('saving network parameters')
                    self.net.save_parameters(os.path.join(file_path, file))

            nd.waitall()

                
from mxnet.io import NDArrayIter

def load_data(dataset_path, skip_test=False, verbose=False):
    """Largely copied from 'readDataset in stardist_mpcdf'"""

    data_classes = ['images', 'masks']
    if skip_test:
        data_purposes = ['train', 'valid']
    else:
        data_purposes = ['train', 'valid', 'test']
        
    dataset_path = Path(dataset_path)

    verbose and print(dataset_path / data_purposes[0] / data_classes[0] / '*.tif')

    X, Y = tuple(
        {data_purpose:
            [imread(str(x)) for x in sorted((dataset_path / data_purpose /  data_class).glob('*.tif'))]
        for data_purpose in data_purposes} for data_class in data_classes)

    return X, Y

    
                
def split_data(idx, num, X, Y):
    
    for s in X.keys():
        if len(X[s]) > num:
            X[s], Y[s] = X[s][idx::num], Y[s][idx::num]
    
    return X, Y


def slice_to_planes(X, Y, step=1, dims=3):
    
    for s in X.keys():
        X[s] = [x[(slice(None), )*j + (i, )] for j in range(dims) for x in  X[s]
                  for i in range(0, x.shape[j], step)]
        Y[s] = [y[(slice(None), )*j + (i, )] for j in range(dims) for y in  Y[s] 
                  for i in range(0, y.shape[j], step)]
   
    return X, Y

def delete_empty_patches(X, Y):
    for s in X.keys():
        sum_Y = [np.sum(y) for y in Y[s]]
        X[s] = [X[s][i] for i in range(len(X[s])) if sum_Y[i] > 0]
        Y[s] = [Y[s][i] for i in range(len(Y[s])) if sum_Y[i] > 0]
    return X, Y

class CellPoseDataIter(DataIter):
    def __init__(self, data, labels, batch_size, num_parts=1, part_index=0, channels=[0,0]):
        self.batch_size = batch_size
        self.cur_iter = 0

                        
        # Data preparation
        print('# reshape data')
        data, _, _ = transforms.reshape_data(data, test_data=None, channels=channels)
        
        if data is None:
            raise ValueError('training data do not all have the same number of channels')
            return
        
        print('# calculate diameters')
        diam = np.array([utils.diameters(label)[0] for label in labels])
        diam[diam<5] = 5.
        
        print('# calculate flows')
        flows = dynamics.labels_to_flows(labels)
        
        self.data = data[part_index::num_parts]
        self.flows = flows[part_index::num_parts]
        self.diam = diam[part_index::num_parts]
        
        self.patch_num = len(self.data)
        self.max_iter = self.patch_num // self.batch_size        
        self.permutation = np.random.permutation(self.patch_num)

    def __iter__(self):
        return self
    
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            
            iter_slice = slice((self.cur_iter-1) * self.batch_size, self.cur_iter * self.batch_size, 1)
            iter_idc = self.permutation[iter_slice]
            
            return list(zip(*[[self.data[i], self.flows[i], self.diam[i]] for i in iter_idc]))
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0
        self.permutation = np.random.permutation(self.patch_num)



if __name__ == '__main__':
    
    # TODO: I notized some differences between the horovod tutorials and my approach:
    # - reset train data (train_data.reset())for every epoch (although the mnist keeps reusing the same pairwise diffent data for all epochs)
    # - both tutorials chose a learning rate which depends on hvd.size() -> Probably needed!
    #   Goya et al.: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
    # - both tutorials use an initalizer for the network weigths (self.net.initalize(initalizer) as well)
    
    
    parser = argparse.ArgumentParser(description='Conduct stardist training with the given input parameter')
    parser.add_argument('dataset_path', metavar='DATASET', type=str, help="dataset path")
    parser.add_argument('model_basedir', metavar='BASEDIR', type=str, help="Output path for all models model")
    parser.add_argument('modelname', metavar='MODELNAME', type=str, help="Name for saving the model")
    parser.add_argument('replicate', metavar='REP', type=int, help='Replicate number')

    args = parser.parse_args()

    verbose = False
    batch_size = 1

    hvd.init()
    num_workers = hvd.size()
    rank = hvd.rank()
    local_rank = hvd.local_rank()

    context = mx.gpu(hvd.local_rank())           
                    
    model = CellposeModel(device=context)
    
    params = model.net.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    X, Y = load_data("datasets/patches-semimanual-raw-64x128x128",
                    skip_test = True, verbose=True)

    verbose and print('num training volumes: ', len(X['train']))

    X, Y = slice_to_planes(X, Y, step=20)
    verbose and print('num training slices: ', len(X['train']))

    X, Y = delete_empty_patches(X, Y)
    verbose and print('num non-empty training slices: ', len(X['train']))

    if hvd.rank() == 0:
        print("num training patches total: {}".format(len(X['train'])))
        print("num validation patches total: {}".format(len(X['valid'])), flush=True)
        

    train_data = CellPoseDataIter(X['train'], Y['train'], batch_size, num_parts=num_workers, part_index=rank)
    # Keep full val_data on each node to make it easier to monitor results
    val_data = CellPoseDataIter(X['valid'], Y['valid'], batch_size)
    
    if hvd.rank() == 0:
        print("num training patches per GPU: {}".format(len(X['train'])))
        print("num validation patches per GPU: {}".format(len(X['valid'])),flush=True)
        
    print(num_workers, local_rank, rank, flush=True)

    modelname = args.modelname + '_rep{}'.format(args.replicate)

    print(modelname)

    model.train(train_data, val_data, n_epochs=500, save_path=str(Path(args.model_basedir) / modelname),
                batch_size=batch_size, params=params)

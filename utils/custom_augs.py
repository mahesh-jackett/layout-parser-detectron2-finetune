from detectron2.data.transforms import Transform, TransformList, NoOpTransform, Augmentation
from albumentations.augmentations.transforms import F as F_
from albumentations.augmentations.transforms import RandomFog, Equalize, ToSepia, InvertImg, RandomShadow
from skimage.util import random_noise as skimage_random_noise
import numpy as np
from typing import Callable
import random
import cv2


shadow = RandomShadow((0,0,1,1), 1, 2, 5, p = 1)

def channel_shuffle(img):
    x = tuple(np.random.randint(0,3,3))
    return F_.channel_shuffle(img, x)

def rgb_shift(image):
    r = random.uniform(-30 , 30)
    g = random.uniform(-30 , 30)
    b = random.uniform(-30 , 30)
    return F_.shift_rgb(image, r, g, b)
    
def fancy_pca(img):
    res = F_.fancy_pca(img, random.gauss(0, 0.9))
    return res

def random_gamma(img):
    gamma = random.uniform(40, 160) / 100.0
    return F_.gamma_transform(img, gamma=gamma)


def random_noise(img):
    mode = random.choice(["gaussian","localvar","poisson","pepper","s&p","speckle"])
    res = skimage_random_noise(img, mode=mode)*255
    return res.astype(np.uint8)


def random_shadow(img):
    try:
        return shadow(image = img)["image"]
    except:
        return img


class BLUR:
    def __init__(self, kernel_size:int = 5):
        self.kernel_size = kernel_size

        self.kernel_h = np.zeros((kernel_size, kernel_size))  # horizontal kernel
        self.kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
        self.kernel_h /= kernel_size 

        self.kernel_v = np.zeros((kernel_size, kernel_size)) # vertical kernel
        self.kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        self.kernel_v /= kernel_size  # Normalize. 

    
    def add_blur(self, img:np.ndarray) -> np.ndarray:
        '''
        Method to add different type of blurs to an image
        args:
            img: Path or the numpy array of image
            kernel_size: Size of the kernel to convolve. Directly dependent on the strength of the blur
            kind: Type of blurring to use. Can be any from ['horizontal_motion','motion_v','average','gauss','median']
        '''
        kind = random.choice(['motion_h','motion_v','average','gauss','median'])
        
        if kind == 'motion_h':
            return cv2.filter2D(img, -1, self.kernel_h) 
    
        elif kind == 'motion_v':
            return cv2.filter2D(img, -1, self.kernel_v)
        
        elif kind == 'average': return cv2.blur(img,(self.kernel_size,self.kernel_size)) # Works like PIL BoxBlur
    
        elif kind == 'gauss': return cv2.GaussianBlur(img, (self.kernel_size,self.kernel_size),0)  
        
        elif kind == 'median': return cv2.medianBlur(img,self.kernel_size) 


class _TransformToAug(Augmentation):
    def __init__(self, tfm: Transform):
        self.tfm = tfm

    def get_transform(self, *args):
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:
        return _TransformToAug(tfm_or_aug)


class KRandomAugmentationList(Augmentation):
    """
    Select and Apply "K" augmentations in "RANDOM" order with "Every"  __call__ method invoke
    """
    def __init__(self, augs, k:int = -1):
        """
        Args:
            augs: list of [Augmentation or Transform]
            k: Number of augment to use from the given list in range [1,len_augs]. If None, use all. If it is -1, generate K randomly between [1,len_augs]
        """
        super().__init__()
        self.max_range = len(augs)
        self.k = k
        self.augs = augs # set augs to use as fixed if we have to use same augs everytime
    

    def _setup_augs(self, augs, k:int):
        '''
        Setup the argument list. Generates the list of argument to use from the given list
        args:
            augs: list of [Augmentation or Transform])
            k: Number of augment to use from the given list in range [1,len_augs]. If False, use all. If it is -1, generate K randomly between [1,len_augs]
        '''
        if k == -1: # Generate a random number
            k = np.random.randint(1,len(augs)+1)
        
        elif k is None: # use all
            k = self.max_range

        temp = np.random.choice(augs,k,replace=False) # get k augments randomly
        return [_transform_to_aug(x) for x in temp]

    
    def __call__(self, aug_input) -> Transform:
        tfms = []

        for x in self._setup_augs(self.augs, self.k): # generate auguments to use randomly on the fly
            tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms)

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return "AugmentationList[{}]".format(", ".join(msgs))

    __str__ = __repr__



class GenericWrapperTransform(Transform):
    """
    Generic wrapper for any transform (for color transform only. You can give functionality to apply_coods, apply_segmentation too)
    """

    def __init__(self, custom_function:Callable):
        """
        Args:
            custom_function (Callable): operation to be applied to the image which takes in an ndarray and returns an ndarray.
        """
        if not callable(custom_function):
            raise ValueError("'custom_function' should be callable")
        
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        '''
        apply transformation to image array based on the `custom_function`
        '''
        return self.custom_function(img)

    def apply_coords(self, coords):
        '''
        Apply transformations to Bounding Box Coordinates. Currently is won't do anything but we can change this based on our use case
        '''
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        '''
        Apply transformations to segmentation. currently is won't do anything but we can change this based on our use case
        '''
        return segmentation


class CustomAug(Augmentation):
    """
    Given a probability and a custom function, return a GenericWrapperTransform object whose `apply_image`  will be called to perform augmentation
    """

    def __init__(self, custom_function, prob=1.0):
        """
        Args:
            custom_op: Operation to use. Must be a function takes an ndarray and returns an ndarray
            prob (float): probability of applying the function
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        '''
        Based on probability, choose whether you want to apply the given function or not
        '''
        do = self._rand_range() < self.prob
        if do:
            return GenericWrapperTransform(self.custom_function)
        else:
            return NoOpTransform() # it returns a Transform which just returns the original Image array only


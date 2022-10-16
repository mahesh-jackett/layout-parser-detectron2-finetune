from detectron2.data.transforms import Transform, TransformList, NoOpTransform, Augmentation
from typing import Any, Callable, List, Optional, TypeVar
from detectron2.data import transforms as T

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
            print(x)
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

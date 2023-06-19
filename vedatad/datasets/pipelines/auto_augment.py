# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import copy

import numpy as np

from vedacore.misc import registry
from .compose import Compose


@registry.register_module('pipeline')
class AutoAugment(object):
    """Auto augmentation.
    这段代码定义了一个名为AutoAugment的类，用于实现自动数据增强。该类是一个数据增强策略的集合，通过随机选择其中的一个策略来增强图像。

    在该类中，定义了以下几个方法和属性：

    __init__(self, policies)：构造方法，接受一个名为policies的参数，表示自动增强的多个策略，其中每个策略由多个数据增强操作组成。在该方法中，对传入的policies进行了类型和格式的校验，并将其深拷贝给self.policies属性。
                              同时，还将每个策略中的多个数据增强操作Compose组合起来，传给self.transforms属性。

    __call__(self, results)：调用方法，接受一个名为results的参数，表示待增强的图像及其相关信息。在该方法中，首先随机选择一个数据增强策略，然后将该策略应用于输入的图像和相关信息，返回增强后的结果。

    __repr__(self)：自定义打印格式。如果打印该类的话，就返回一个字符串表示该类的信息，包括类名和策略。

    此外，代码中还定义了一个装饰器@registry.register_module('pipeline')，用于将该类注册为pipeline模块的一个组件。

    最后提供了一个示例，展示了如何使用该类进行自动数据增强，包括实例化一个AutoAugment对象、调用该对象对输入数据进行增强的步骤。
    
    This data augmentation is proposed in `Learning Data Augmentation
    Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_.

    TODO: Implement 'Shear', 'Sharpness' and 'Rotate' transforms

    Args:
        policies (list[list[dict]]): The policies of auto augmentation. Each
            policy in ``policies`` is a specific augmentation policy, and is
            composed by several augmentations (dict). When AutoAugment is
            called, a random policy in ``policies`` will be selected to
            augment images.

    Examples:
        >>> replace = (104, 116, 124)
        >>> policies = [
        >>>     [
        >>>         dict(type='Sharpness', prob=0.0, level=8),
        >>>         dict(
        >>>             type='Shear',
        >>>             prob=0.4,
        >>>             level=0,
        >>>             replace=replace,
        >>>             axis='x')
        >>>     ],
        >>>     [
        >>>         dict(
        >>>             type='Rotate',
        >>>             prob=0.6,
        >>>             level=10,
        >>>             replace=replace),
        >>>         dict(type='Color', prob=1.0, level=6)
        >>>     ]
        >>> ]
        >>> augmentation = AutoAugment(policies)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self, policies):
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'

        self.policies = copy.deepcopy(policies) 
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        return transform(results)

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies}'

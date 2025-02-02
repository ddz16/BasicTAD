# adapted from https://github.com/open-mmlab/mmcv
import inspect

from .utils import is_str


class Registry:
    """A registry to map strings to classes.
    
    这个注册器有两个层次，即两层字典。第一个层次是module_name，第二个层次是class_name。这在get函数中可以看出来。
    比如module_name为dataset时，self._module_dict['dataset']还是一个字典，包含类名到类的映射，比如'Thumos'->Thumos类，'Activity'->Activity类。
    
    __new__方法实现了单例模式，保证只有一个Registry实例。
    
    __len__方法返回注册表中注册的类的数量。
    
    __contains__方法判断给定的key是否在注册表中。
    
    get方法根据给定的类名和模块名返回对应的类。
    
    _register_module方法将给定的类和模块名注册到注册表中。
    
    register_module方法是一个装饰器，用于将类注册到注册表中。

    """
    _instance = None

    def __init__(self):
        self._module_dict = dict()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(items={self._module_dict})'
        return format_str

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, cls_name, module_name='module'):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        if module_name not in self._module_dict:
            raise KeyError(f'{module_name} is not in registry')
        dd = self._module_dict[module_name]
        # print("module_name:",module_name)
        # print("dd:",dd)
        if cls_name not in dd:
            raise KeyError(f'{cls_name} is not registered in {module_name}')

        return dd[cls_name]

    def _register_module(self, cls, module_name):
        if not inspect.isclass(cls):
            raise TypeError('module must be a class, ' f'but got {type(cls)}')

        cls_name = cls.__name__
        self._module_dict.setdefault(module_name, dict())
        dd = self._module_dict[module_name]
        if cls_name in dd:
            raise KeyError(f'{cls_name} is already registered '
                           f'in {module_name}')
        dd[cls_name] = cls

    def register_module(self, module_name='module'):

        def _register(cls):
            self._register_module(cls, module_name)
            return cls

        return _register


registry = Registry()


def build_from_cfg(cfg, registry, module_name='module', default_args=None):
    ''' 根据给定的配置字典从注册表中构建一个类的实例，类由cfg字典中的typename指定，类的初始化参数由cfg字典中其他键值对指定。它具有以下参数：
        cfg：配置字典，必须包含键"typename"，指定要构建的类。
        registry：注册表对象，用于查找类。
        module_name：模块名，默认为"module"。
        default_args：默认参数字典，用于设置类的默认参数。
    '''
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'typename' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "typename", but got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be a registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()
    obj_type = args.pop('typename')
    if is_str(obj_type):
        obj_cls = registry.get(obj_type, module_name)
    else:
        raise TypeError(f'type must be a str, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():  # 遍历default_args字典中的键值对，并将键值对添加到args字典中。如果args字典中已经存在相同的键，则不会覆盖原有的值。
            args.setdefault(name, value)  
    return obj_cls(**args)


def build_from_module(cfg, module, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'typename' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "typename", but got {cfg}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()
    obj_type = args.pop('typename')
    if is_str(obj_type):
        obj_cls = getattr(module, obj_type)
    else:
        raise TypeError(f'type must be a str, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():  # 遍历default_args字典中的键值对，并将键值对添加到args字典中。如果args字典中已经存在相同的键，则不会覆盖原有的值。
            args.setdefault(name, value)
    return obj_cls(**args)

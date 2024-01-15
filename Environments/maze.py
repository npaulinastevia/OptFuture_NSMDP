from abc import ABC
from abc import abstractmethod

from collections import namedtuple
import numpy as np

from .object import Object


class BaseMaze(ABC):
    def __init__(self, **kwargs):
        self.objects = self.make_objects()
        assert all([isinstance(obj, Object) for obj in self.objects])

        db={}
        for x in self.objects:
            db[x.name]=x
        self.objects=db

        #objs = namedtuple('Objects', map(lambda x: x.name, self.objects))
        #objs.__new__.__defaults__ = self.objects
        #self.objects = objs()

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    @abstractmethod
    def size(self):
        r"""Returns a pair of (height, width). """
        pass
        
    @abstractmethod
    def make_objects(self):
        r"""Returns a list of defined objects. """
        pass
    
    def _convert(self, x, name):

        for key,obj in self.objects.items():
            pos = np.asarray(obj.positions)
            x[pos[:, 0], pos[:, 1]] = getattr(obj, name, None)
        return x
    
    def to_name(self):
        x = np.empty(self.size, dtype=object)
        return self._convert(x, 'name')
    
    def to_value(self):
        x = np.empty(self.size, dtype=np.float32)
        return self._convert(x, 'value')
    
    def to_rgb(self):
        x = np.empty((*self.size, 3), dtype=np.uint8)
        return self._convert(x, 'rgb')
    
    def to_impassable(self):
        x = np.empty(self.size, dtype=bool)
        return self._convert(x, 'impassable')
    
    def __repr__(self):
        return f'{self.__class__.__name__}{self.size}'

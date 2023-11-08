from abc import ABC, abstractmethod
import cv2


class Shape(ABC):
    @abstractmethod
    def draw(self, image):
        pass


class Rect:
    def __init__(self, xywh):
        self._xywh = xywh

    def draw(self, image):
        box = self._xywh
        cv2.rectangle(image, box[0:2], box[0:2] + box[2:], (0, 0, 255), 2)

    @property
    def rect(self):
        return self._xywh


class Landmarks:
    def __init__(self, coords=None):
        self._coords = coords[0]

    def draw(self, image):
        if self._coords is None:
            return

        for x,y in self._coords:
            x, y = round(x), round(y)
            cv2.circle(image, (x, y), 1, (0, 255, 0), 2)


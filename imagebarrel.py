import numpy as np

class ImageBarrel:
    def __init__(self, size: int):
        if size <= 1:
            raise ValueError("size must be larger than one")
        self.size = size
        self.tensor = None
        self.index = 0
        self.full = False

    def append(self, img):
        if self.tensor is None:
            w, h, c = img.shape   # check rgb image. only works for rgb image currently.
            assert c == 3

            self.tensor = np.zeros((*img.shape, self.size), dtype=img.dtype)
            
        self.tensor[..., self.index] = img
        self.index = self.index + 1
        if self.index >= self.size:
            self.full = True
            self.index = 0

    def clear(self):
        self.tensor = None
        self.index = 0
        self.full = False

    def range(self, n: int):
        if n <= 0 or n > self.size:
            n = self.size
        end_index = self.index
        start_index = (self.index - n) % self.size
        if start_index < end_index:
            max_img = np.max(self.tensor[..., start_index:end_index], axis=-1)
            min_img = np.min(self.tensor[..., start_index:end_index], axis=-1)
        elif start_index == end_index:
            max_img = np.max(self.tensor, axis=-1)
            min_img = np.min(self.tensor, axis=-1)
        else:
            max_img = np.maximum(np.max(self.tensor[..., :end_index]), np.max(self.tensor[..., start_index:]))
            min_img = np.minimum(np.min(self.tensor[..., :end_index]), np.min(self.tensor[..., start_index:]))
        diff_img = max_img - min_img
        if len(diff_img.shape) == 3:
            diff_img = np.max(diff_img, axis=-1) # here, axis=-1 is the color axis
        elif len(diff_img.shape) == 2:
            pass
        else:
            raise ValueError("unsupported image type")
        return np.mean(diff_img) / 255.0

    def median(self, n: int=0):
        if not self.full:
            assert False

        if n <= 0 or n > self.size:
            n = self.size

        end_index = self.index
        start_index = (self.index - n) % self.size
        if start_index < end_index:
            tensor_slice = self.tensor[..., start_index:end_index]
        elif start_index == end_index:
            tensor_slice = self.tensor
        else:
            tensor_slice = np.concatenate(self.tensor[..., start_index:], self.tensor[..., :end_index], axis=-1)
        return np.median(tensor_slice, axis=-1).astype(np.uint8)


class ImageAggregate:
    def __init__(self, size: int):
        self.size = size
        self.items = []

    def append(self, img):
        if len(self.items) < self.size:
            self.items.append(img)

    def __len__(self) -> int:
        return len(self.items)

    def median(self):
        arr = np.array(self.items)
        output = np.median(arr, axis=0).astype(np.uint8)
        return output
    
    def clear(self):
        self.items.clear()


import random
import math
from torchvision import transforms
import torch 

def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    org_imgs, imgs = list(zip(*batch))
#     print(list(zip(*batch)))
    return torch.stack(org_imgs), torch.stack(imgs)

class CutPaste(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, colorJitter=0.1, transform=None):
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.transform = transform
        
        # setup colorJitter
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter)

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # TODO: check if this is realy uniform in (aspect_ratio, 1) ∪ (1, 1/aspect_ratio).
        # so first we sample from witch bucket and than in the range
        aspect = random.uniform(self.aspect_ratio, 1/self.aspect_ratio)
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # BIG TODO: also sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        org_img = img.copy()
        img.paste(patch, insert_box)
        
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        
        return org_img, img
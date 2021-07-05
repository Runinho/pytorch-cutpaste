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
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform
        
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter)
    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img
    
class CutPasteNormal(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.transform = transform

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # TODO: check if this is realy uniform in (aspect_ratio, 1) ∪ (1, 1/aspect_ratio).
        # so first we sample from witch bucket and than in the range
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()
        
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
        
        return org_img, img

class CutPasteScar(CutPaste):
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation
    
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg,expand=True)
        
        #paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        insert_box = [to_location_w, to_location_h, to_location_w + patch.size[1], to_location_h + patch.size[0]]
        mask = patch.split()[-1]
        patch = patch.convert("RGB")
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        org_img = img.copy()
        img.paste(patch, (to_location_w, to_location_h), mask=mask)
        
        return super().__call__(org_img, img)
    
class CutPasteUnion(object):
    
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)
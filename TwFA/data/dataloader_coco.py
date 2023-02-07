import json, os, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import random
from scipy import misc
import albumentations


class CocoDataset(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json=None,
                 stuff_only=True, size=296, crop_size=256, emb_size=16, mask_size=16,
                 normalize_images=True, max_samples=None,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None,
                 crop_type='random', class_num=184, mask_inst_only=False):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.
    
        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        - size: resize the image, the longest edge's size.
        - crop_size: the cropped image patch's size: crop_size*crop_size.
        - crop_type: (1) 'random', randomly crop a crop_size*crop_size patch;
                     (2) 'none', only resize image to size*size, then crop a 
                          crop_size*crop_size patch;
                     (3) 'center' crop a crop_size*crop_size patch in the 
                          center of the image.
        - mask_inst_only: build connection matrix only for instance objects.
        """
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.size = size
        self.crop_size = crop_size
        self.emb_size = emb_size
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.max_objects_per_image = max_objects_per_image
        self.normalize_images = normalize_images
        self.include_relationships = include_relationships
        self.left_right_flip = left_right_flip
        self.class_num = class_num
        self.crop_type = crop_type
        self.min_object_size = min_object_size
        self.mask_inst_only = mask_inst_only

        # Assign random seed
        random.seed(44)


        if crop_type == 'none':
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
        else:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                    self.image_id_to_objects[image_id].append(object_data)

            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)


        self.image_ids = new_image_ids



        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

        print('total image number', self.__len__())


    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.
    
        Returns a dict of:
        - image: FloatTensor of shape (C, H, W)
        - label: LongTensor of shape (O,)
        - bbox: FloatTensor of shape (O, 8), the first four numbers gives boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system, the last four are the tokenized ones.
        - mask: LongTensor of shape (M, M) giving the connection matrix.
        """
        # Decide whether to flip
        flip = False
        if random.random() < 0.5 and self.left_right_flip:
            flip = True
        image_id = self.image_ids[index]

        # Process image
        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                image = np.array(image).astype(np.uint8)
                image = self.rescaler(image=image)['image']

                HH_new, WW_new, _ = image.shape
                
                crop_size = self.crop_size
                if self.crop_type == 'center':
                    x_shift = int(WW_new - crop_size) // 2
                    y_shift = int(HH_new - crop_size) // 2
                else:
                    x_shift = random.randint(0, WW_new - crop_size)
                    y_shift = random.randint(0, HH_new - crop_size)
                image = image[y_shift: y_shift + crop_size, x_shift: x_shift + crop_size]
                image = albumentations.Resize(height=self.crop_size, width=self.crop_size)(image=image)['image']
                image = (image / 127.5 - 1.0).astype(np.float32)
                image = torch.from_numpy(image)  # h, w, c
                image = image.permute(2, 0, 1)

        # Prepare label
        objs, boxes = [], []
        label_mask = torch.zeros(self.emb_size**2, self.max_objects_per_image)
        token_mask = torch.zeros(self.emb_size**2, self.emb_size, self.emb_size)
        label_n = 0

        for o_i, object_data in enumerate(self.image_id_to_objects[image_id]):
            x, y, w, h = object_data['bbox']
            x0 = x / WW * WW_new
            y0 = y / HH * HH_new
            x1 = (w) / WW * WW_new
            y1 = (h) / HH * HH_new
            if flip:
                x0 = WW_new - (x0 + x1)
            if x0 + x1 <= x_shift or y0 + y1 <= y_shift or x0 >= x_shift + crop_size or y0 >= y_shift + crop_size:
                continue

            x0_new = max(x0 - x_shift, 0)
            y0_new = max(y0 - y_shift, 0)
            x1_new = min(x0 + x1 - x_shift, crop_size) - x0_new
            y1_new = min(y0 + y1 - y_shift, crop_size) - y0_new

            x0 = x0_new / crop_size
            y0 = y0_new / crop_size
            x1 = x1_new / crop_size
            y1 = y1_new / crop_size

            if x1 * y1 <= self.min_object_size or x1 < 0.1 or y1 < 0.1:
                continue
            
            x_pos0 = round(x0 * (self.crop_size-1) + 1)
            x_pos1 = round((x0+x1) * (self.crop_size-1) + 1)
            y_pos0 = round(y0 * (self.crop_size-1) + 1)
            y_pos1 = round((y0+y1) * (self.crop_size-1) + 1)

            boxes.append(np.array([x0, y0, x1, y1, x_pos0, x_pos1, y_pos0, y_pos1]))
            objs.append(object_data['category_id'])

            if self.mask_inst_only:
                if object_data['category_id'] < 91:
                    build_mask = True
                else:
                    build_mask = False
            else:
                build_mask = True

            if build_mask:
                tmp_mask_token = torch.zeros(self.emb_size, self.emb_size)
                x_pos0 = round(x0 * (self.emb_size-1))
                x_pos1 = round((x0+x1) * (self.emb_size-1))
                y_pos0 = round(y0 * (self.emb_size-1))
                y_pos1 = round((y0+y1) * (self.emb_size-1))
                tmp_mask_token[y_pos0:y_pos1+1, x_pos0:x_pos1+1] = 1
                tmp_mask_token = tmp_mask_token.view(-1)
                tmp_index = (tmp_mask_token==1)
                label_mask[tmp_index, label_n] = 1
                token_mask[:, y_pos0:y_pos1+1, x_pos0:x_pos1+1] = 1
            label_n += 1
        

        # Add 0 for number of objects
        for o_i in range(len(objs), self.max_objects_per_image):
            objs.append(self.vocab['object_name_to_idx']['__image__'])
            boxes.append(np.array([0.0, 0.0, 1.0, 1.0, 0, 0, 0, 0]))



        objs = torch.LongTensor(objs)
        boxes = np.vstack(boxes)

        
        
        # Building connectivity matrix
        tmp_index = label_mask.sum(-1)==0
        label_mask[tmp_index] = 1
        token_mask[tmp_index] = token_mask[tmp_index] + 1
        token_mask[token_mask > 1] = 0
        tmp_label_mask = torch.cat([label_mask.unsqueeze(-1), label_mask.unsqueeze(-1)], dim=-1).reshape(label_mask.shape[0], -1)
        mask = torch.cat([label_mask, tmp_label_mask, tmp_label_mask, token_mask.view(token_mask.shape[0], -1)], dim=-1).contiguous()
        tmp_mask = torch.ones((self.max_objects_per_image*5, self.max_objects_per_image*5+self.emb_size**2))
        mask = torch.cat([tmp_mask, mask], dim=0)
        tmp_mask = torch.tril(torch.ones(self.emb_size**2 + self.max_objects_per_image*5,
                                     self.emb_size**2 + self.max_objects_per_image*5))
        mask = mask * tmp_mask
        mask[:self.max_objects_per_image*5, :self.max_objects_per_image*5] = 1
        
        # Output data
        out_data = {}
        out_data['mask'] = mask
        out_data['image'] = image
        out_data['label'] = objs
        out_data['bbox'] = torch.from_numpy(boxes)

        return out_data





def seg_to_mask(seg, width=1.0, height=1.0):
    """
    Tiny utility for decoding segmentation masks using the pycocotools API.
    """
    if type(seg) == list:
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle)


def coco_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:
  
    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    """
    all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, masks, triples) in enumerate(batch):
        all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out


# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=True):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images
    
    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255).byte()
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


def unpack_var(v):
    if isinstance(v, torch.autograd.Variable):
        return v.data
    return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
    triples = unpack_var(triples)
    obj_data = [unpack_var(o) for o in obj_data]
    obj_to_img = unpack_var(obj_to_img)
    triple_to_img = unpack_var(triple_to_img)

    triples_out = []
    obj_data_out = [[] for _ in obj_data]
    obj_offset = 0
    N = obj_to_img.max() + 1
    for i in range(N):
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)

        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        triples_out.append(cur_triples)

        for j, o_data in enumerate(obj_data):
            cur_o_data = None
            if o_data is not None:
                cur_o_data = o_data[o_idxs]
            obj_data_out[j].append(cur_o_data)

        obj_offset += o_idxs.size(0)

        return triples_out, obj_data_out

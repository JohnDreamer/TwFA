import os
import random
from collections import defaultdict

import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage.transform import resize as imresize

import numpy as np
import h5py
import PIL
import albumentations

# Split classes into instance and stuff
inst_name = ['man', 'head', 'leg', 'woman', 'eye', 'shoe', 'people', 'wheel', 'handle', 'nose', 'arm', 'plate', 'hat', 'tail', 'foot', 'face', 'boy', 'helmet', 'tire', 'girl', 'mouth', 'cap', 'glasses', 'headlight', 'trunk', 'neck', 'flag', 'seat', 'wing', 'lamp', 'finger', 'bike', 'animal', 'sunglass', 'container', 'knob', 'boot', 'paw', 'horn', 'engine', 'child', 'player', 'lady', 'vehicle', 'basket', 'person', 'sign', 'light', 'head', 'hand', 'hair', 'car', 'ear', 'table', 'chair', 'bag', 'snow', 'bottle', 'glass', 'boat', 'plant', 'umbrella', 'bird', 'banana', 'bench', 'book', 'top', 'wave', 'clock', 'glove', 'bowl', 'bus', 'train', 'horse', 'kite', 'board', 'cup', 'elephant', 'giraffe', 'cow', 'dog', 'sheep', 'zebra', 'ski', 'ball', 'back', 'truck', 'sand', 'skateboard', 'motorcycle', 'cat', 'bed', 'sink', 'surfboard', 'pizza', 'bear', 'orange', 'pot', 'apple', 'plane', 'key', 'tie']
stuff_name = ['__image__', 'window', 'shirt', 'leaf', 'pole', 'line', 'shadow', 'letter', 'pant', 'stripe', 'jacket', 'number', 'sidewalk', 'short', 'spot', 'street', 'logo', 'background', 'post', 'picture', 'button', 'track', 'part', 'box', 'jean', 'edge', 'reflection', 'writing', 'coat', 'sock', 'word', 'wire', 'frame', 'windshield', 'beach', 'photo', 'hole', 'stand', 'ocean', 'sticker', 'design', 'window', 'tree', 'wall', 'building', 'ground', 'light', 'sky', 'grass', 'car', 'cloud', 'table', 'door', 'flower', 'water', 'fence', 'floor', 'rock', 'road', 'tile', 'snow', 'bush', 'field', 'roof', 'branch', 'plant', 'brick', 'food', 'dirt', 'mirror', 'pillow', 'shelf', 'bus', 'paper', 'house', 'board', 'cup', 'cabinet', 'mountain', 'counter', 'hill', 'ceiling', 'sand', 'curtain', 'towel', 'wood', 'stone', 'railing']


class VgDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, size=296, crop_size=256, emb_size=16,
                 normalize_images=True, max_objects=30, min_object_size=0.02,
                 include_relationships=True, use_orphaned_objects=True,
                 left_right_flip=False, crop_type='random',
                 mask_inst_only=False):
        super(VgDataset, self).__init__()

        vocab = json.load(open(vocab, 'r'))
        inst_idx = []
        for name in inst_name:
            inst_idx.append(vocab['object_name_to_idx'][name])
        self.inst_idx = inst_idx

        self.image_dir = image_dir
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.left_right_flip = left_right_flip
        self.include_relationships = include_relationships
        self.mask_inst_only = mask_inst_only
        self.crop_type = crop_type
        self.size = size
        self.crop_size = crop_size
        self.emb_size = emb_size
        self.min_object_size = min_object_size

        # Assign random seed
        random.seed(44)

        if crop_type == 'none':
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
        else:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))


        print('total image number', self.__len__())


    def __len__(self):
        num = self.data['object_names'].size(0)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
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
        img_path = os.path.join(self.image_dir, self.image_paths[index].decode('UTF-8'))

        with open(img_path, 'rb') as f:
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

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects and self.use_orphaned_objects:
            num_to_add = self.max_objects - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        
        objs, boxes = [], []

        label_mask = torch.zeros(self.emb_size**2, self.max_objects)
        token_mask = torch.zeros(self.emb_size**2, self.emb_size, self.emb_size)
        label_n = 0

        for i, obj_idx in enumerate(obj_idxs):
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW * WW_new
            y0 = float(y) / HH * HH_new
            x1 = float(w) / WW * WW_new
            y1 = float(h) / HH * HH_new
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
            objs.append(self.data['object_names'][index, obj_idx].item())
            
            if self.mask_inst_only:
                if self.data['object_names'][index, obj_idx].item() in self.inst_idx:
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

        
        for i in range(len(objs), self.max_objects):
            objs.append(self.vocab['object_name_to_idx']['__image__'])
            boxes.append(np.array([0.0, 0.0, 1.0, 1.0, 0, 0, 0, 0]))
        
        objs = torch.LongTensor(objs)
        boxes = np.vstack(boxes)

        # Process mask
        tmp_index = label_mask.sum(-1)==0
        label_mask[tmp_index] = 1
        token_mask[tmp_index] = token_mask[tmp_index] + 1
        token_mask[token_mask > 1] = 0
        tmp_label_mask = torch.cat([label_mask.unsqueeze(-1), label_mask.unsqueeze(-1)], dim=-1).reshape(label_mask.shape[0], -1)
        mask = torch.cat([label_mask, tmp_label_mask, tmp_label_mask, token_mask.view(token_mask.shape[0], -1)], dim=-1).contiguous()
        tmp_mask = torch.ones((self.max_objects*5, self.max_objects*5+self.emb_size**2))
        mask = torch.cat([tmp_mask, mask], dim=0)
        tmp_mask = torch.tril(torch.ones(self.emb_size**2 + self.max_objects*5,
                                         self.emb_size**2 + self.max_objects*5))
        mask = mask * tmp_mask
        mask[:self.max_objects*5, :self.max_objects*5] = 1

        # Output data
        out_data = {}
        out_data['mask'] = mask
        out_data['image'] = image
        out_data['label'] = objs
        out_data['bbox'] = torch.from_numpy(boxes)

        return out_data


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


IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:
    
    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
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
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out


def vg_uncollate_fn(batch):
    """
    Inverse operation to the above.
    """
    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
    out = []
    obj_offset = 0
    for i in range(imgs.size(0)):
        cur_img = imgs[i]
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)
        cur_objs = objs[o_idxs]
        cur_boxes = boxes[o_idxs]
        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        obj_offset += cur_objs.size(0)
        out.append((cur_img, cur_objs, cur_boxes, cur_triples))
    return out

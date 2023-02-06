
import yaml
import torch
from omegaconf import OmegaConf
from TwFA.modules.taming.models.vqgan import VQModel
from TwFA.models.cond_transformer_twfa import Net2NetTransformer
from TwFA.data.dataloader_coco import CocoDataset
import os
from tqdm import tqdm
# from scipy import misc
import imageio
from collections import OrderedDict
import numpy as np
from PIL import Image
import argparse
import torch.nn.functional as F


def masks_to_layout(boxes, masks, H, W=None):
    """
    Inputs:
        - boxes: Tensor of shape (b, num_o, 4) giving bounding boxes in the format
            [x0, y0, x1, y1] in the [0, 1] coordinate space
        - masks: Tensor of shape (b, num_o, M, M) giving binary masks for each object
        - H, W: Size of the output image.
    Returns:
        - out: Tensor of shape (N, num_o, H, W)
    """
    b, num_o, _ = boxes.size()
    M = masks.size(3)
    C = masks.size(2)
    assert masks.size() == (b, num_o, C, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes.view(b*num_o, -1), H, W).float().to(masks.device)

    img_in = masks.float().view(b*num_o, C, M, M)
    sampled = F.grid_sample(img_in, grid, mode='nearest')

    return sampled.view(b, num_o, C, H, W)


def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output
    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2], boxes[:, 3]

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None):
  model = VQModel(**config.model.params.first_stage_config.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in sd.items():
        if 'first_stage_model' in k:
            name = k[18:]  # remove `module.`nvidia
            new_state_dict[name] = v

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def load_transformer(config, ckpt_path=None):
    model = Net2NetTransformer(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()


def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(args, model, x, label, bbox, steps, flag=None, mask=None, temperature=1.0, sampled=True):
    for k in range(steps):
        x_cond = x # if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        if mask is not None:
            logits, _ = model.transformer(x_cond, label, bbox, mask)
        else:
            logits, _ = model.transformer(x_cond, label, bbox)

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        logits_obj = top_k_logits(logits, args.topk_obj)
        logits_bg = top_k_logits(logits, args.topk_bg)

        # apply softmax to convert to probabilities
        probs_obj = F.softmax(logits_obj, dim=-1)
        probs_bg = F.softmax(logits_bg, dim=-1)

        # sample from the distribution or take the most likely
        if sampled:
            if flag is not None:
                ix_obj = torch.multinomial(probs_obj, num_samples=1)
                ix_bg = torch.multinomial(probs_bg, num_samples=1)
                ix = ix_obj * flag[:, k-steps].unsqueeze(1) + ix_bg * (1-flag[:, k-steps]).unsqueeze(1)
                ix = ix.long()
            else:
                ix = torch.multinomial(probs_bg, num_samples=1)
        else:
            _, ix = torch.topk(probs_bg, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    return x

# @torch.no_grad()
def run_conditional(model, dsets, args):
    sample_save_path = args.save_path
    
    # make the save dir
    os.makedirs(sample_save_path, exist_ok=True)

    n = 0

    with torch.no_grad():
        for data in tqdm(dsets):
            # processing data
            img = data['image'].cuda()
            label = data['label'].cuda()
            bbox = data['bbox'].float().cuda()
            mask = data['mask'].cuda()
            b, o = label.shape[:2]
            mask_new = torch.ones((b, o, 1, 16, 16), device=label.device)
            mask_new = masks_to_layout(bbox[:, :, :4].contiguous(), mask_new, 16, 16)
            mask_new = mask_new.reshape(b, o, -1)
            mask_new = mask_new * (label <= 91).float().view(b, o, 1)
            mask_new = (mask_new.sum(dim=1) > 0).float()
            
            # inference
            quant_z, z_indices = model.encode_to_z(img)
            x = z_indices[:, :0].long()
            index_sample = sample(args, model, x, label, bbox, 256, flag=mask_new, mask=mask)
            x_sample_nopix = model.decode_to_img(index_sample, quant_z.shape)

            # save images
            for i in range(img.shape[0]):
                imageio.imwrite("{save_path}/sam_{s_i}.png".format(save_path=sample_save_path, s_i=i + n),
                                custom_to_pil(x_sample_nopix[i]))
            n += img.shape[0]

    print('finish !')


def main(args):
    config_path = args.base
    model_path = args.path
    config = load_config(config_path, display=False)
    model = load_transformer(config, ckpt_path=model_path).cuda().eval()

    dataset = CocoDataset(size=256,
            crop_size=256,
            image_dir='data/coco/train2017/',
            instances_json='data/coco/annotations/instances_train2017.json',
            stuff_json='data/coco/annotations/stuff_train2017.json',
            stuff_only=True, mask_inst_only=True,
            left_right_flip=False, crop_type='none')

    print('total image num', len(dataset))

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            drop_last=False, shuffle=False, num_workers=8)


    run_conditional(model, dataloader, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/coco.yaml',
                        help='config file')
    parser.add_argument('--path', type=str, default='pretrained/checkpoints/TwFA_ckpt_coco.ckpt',
                        help='checkpoint path')
    parser.add_argument('--save_path', type=str, default='samples/coco',
                        help='dir path to save')
    parser.add_argument('--topk_obj', type=int, default=5,
                        help='the topk of object regions')
    parser.add_argument('--topk_bg', type=int, default=20,
                        help='the topk of background regions')
    args = parser.parse_args()
    main(args)

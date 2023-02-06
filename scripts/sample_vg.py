
import yaml
import torch
from omegaconf import OmegaConf
from TwFA.modules.taming.models.vqgan import VQModel
from TwFA.models.cond_transformer_twfa import Net2NetTransformer
from TwFA.data.dataloader_vg import VgDataset
import os
from tqdm import tqdm
# from scipy import misc
import imageio
from collections import OrderedDict
import numpy as np
from PIL import Image
import argparse


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

# @torch.no_grad()
def run_conditional(model, dsets, args):
    sample_save_path = args.save_path

    os.makedirs(sample_save_path, exist_ok=True)

    n = 0

    with torch.no_grad():
        for data in tqdm(dsets):
            if n + data['image'].shape[0] < args.start:
                n += data['image'].shape[0]
                continue
            if args.end > -1 and n > args.end:
                break
            img = data['image'].cuda()
            label = data['label'].cuda()
            bbox = data['bbox'].float().cuda()
            mask = data['mask'].cuda()

            x_sample_nopix = model(img, label, bbox, mask, mode='val')
            print(n, 'images have been generated')
            for i in range(img.shape[0]):
                imageio.imwrite("{save_path}/sam_{s_i}.jpg".format(save_path=sample_save_path, s_i=i + n),
                                custom_to_pil(x_sample_nopix[i]))
            n += img.shape[0]

    print('finish !')


def main(args):
    config_path = args.base
    model_path = args.path
    config = load_config(config_path, display=False)
    model = load_transformer(config, ckpt_path=model_path).cuda().eval()

    dataset = VgDataset(size=256,
            crop_size=256,
            image_dir='data/vg_other_annotations/images/',
            vocab='data/vg_other_annotations/vocab.json',
            h5_path='data/vg_other_annotations/test.h5',
            mask_inst_only=True, left_right_flip=False, crop_type='none'
            )

    print('total image num', len(dataset))

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            drop_last=False, shuffle=False, num_workers=8)


    run_conditional(model, dataloader, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/vg.yaml',
                        help='config file path')
    parser.add_argument('--path', type=str, default='pretrained/checkpoints/TwFA_ckpt_vg.ckpt',
                        help='checkpoint path')
    parser.add_argument('--save_path', type=str, default='samples/vg',
                        help='dir path to save')
    parser.add_argument('--topk_obj', type=int, default=100,
                        help='the topk of object regions')
    parser.add_argument('--topk_bg', type=int, default=100,
                        help='the topk of background regions')
    args = parser.parse_args()
    main(args)

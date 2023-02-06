import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn


from main import instantiate_from_config
from collections import OrderedDict

# add layout encoding of High-Resolution Complex Scene Synthesis with Transformer


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
    sampled = F.grid_sample(img_in, grid, mode='bilinear')

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


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self, transformer_config, first_stage_config,
                 permuter_config=None,
                 pretrain_path=None,
                 ckpt_path=None, ignore_keys=[],
                 first_stage_key="image",
                 label_key='label',
                 bbox_key='bbox',
                 vqvae_ckpt_path=None,
                 use_vqgan=True,
                 pkeep=1.0,
                 img_token_size=16,
                 ):

        super().__init__()
        self.use_vqgan = use_vqgan
        self.init_first_stage_from_ckpt(first_stage_config, vqvae_ckpt_path)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if pretrain_path is not None:
            self.init_from_pretrained_ckpt(pretrain_path)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.first_stage_key = first_stage_key
        self.label_key = label_key
        self.bbox_key = bbox_key
        self.pkeep = pkeep

        self.img_token_size = img_token_size


    def init_from_pretrained_ckpt(self, path):
        sd = torch.load(path, map_location='cpu')['state_dict']
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in sd.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Restored from {path} pretrained model")


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config, ckpt_path):
        model = instantiate_from_config(config)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            sd = state_dict['state_dict']
            new_state_dict = OrderedDict()
            only_use_vqgan = True  
            for k, v in sd.items():
                if 'first_stage_model' in k:
                    only_use_vqgan = False
                    break
            for k, v in sd.items():
                if only_use_vqgan:
                    new_state_dict[k] = v
                else:
                    if 'first_stage_model' in k:
                        name = k[18:]  # remove first_stage_model.
                        new_state_dict[name] = v

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print('error: not found vqvae ckpt')
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    

    def forward(self, x, label, bbox, mask, mode='train'):
        if mode == 'train':
            # one step to produce the logits
            _, z_indices = self.encode_to_z(x)

            if self.training and self.pkeep < 1.0:
                mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                             device=z_indices.device))
                mask = mask.round().to(dtype=torch.int64)
                r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
                a_indices = mask*z_indices+(1-mask)*r_indices
            else:
                a_indices = z_indices


            # target includes all sequence elements (no need to handle first one
            # differently because we are conditioning)
            target = z_indices
            # make the prediction
            logits, loss = self.transformer(a_indices[:, :-1], label, bbox, mask)

            return logits, target, loss
        else:
            quant_z, z_indices = self.encode_to_z(x)
            z_start_indices = z_indices[:, :0]
            index_sample = self.sample(z_start_indices, label, bbox, mask,
                                       steps=z_indices.shape[1],
                                       temperature=1.0,
                                       sample=True,
                                       top_k=100,
                                       callback=lambda k: None)
            x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
            return x_sample_nopix

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, label, bbox, mask, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):

        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = label.clone()[:,x.shape[1]-label.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x, label, bbox, mask)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond, label, bbox, mask)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
        return x


    @torch.no_grad()
    def encode_to_z(self, x):
        if self.use_vqgan:
            quant_z, _, info = self.first_stage_model.encode(x)
            indices = info[2].view(quant_z.shape[0], -1)
            indices = self.permuter(indices)
            return quant_z, indices
        else:
            quant_z, _, info = self.first_stage_model.encode(x)
            indices = info.view(quant_z.shape[0], -1)
            indices = self.permuter(indices)
            return quant_z, indices    # b, c, h, w     b, h*w

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        if self.use_vqgan:
            index = self.permuter(index, reverse=True)
            bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
            quant_z = self.first_stage_model.quantize.get_codebook_entry(
                index.reshape(-1), shape=bhwc)
            x = self.first_stage_model.decode(quant_z)
            return x
        else:
            # zshape: b,c,h,w
            b, c, h, w = zshape
            index = index.view((b, h, w))
            x = self.first_stage_model.decode_code(index)
            return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, label, bbox, mask = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, label, bbox, mask = self.get_xc(batch, N)
        x = x.to(device=self.device)
        label = label.to(device=self.device)
        bbox = bbox.to(device=self.device)
        mask = mask.to(device=self.device)


        quant_z, z_indices = self.encode_to_z(x)
        c_indices = label

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c_indices, bbox, mask,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices, bbox, mask,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices, bbox, mask,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        label = self.get_input(self.label_key, batch)
        bbox = self.get_input(self.bbox_key, batch).float()
        mask = self.get_input('mask', batch)

        label = label
        if N is not None:
            x = x[:N]
            label = label[:N]
            bbox = bbox[:N]
            mask = mask[:N]

        return x, label, bbox, mask

    def shared_step(self, batch):
        x, label, bbox, mask = self.get_xc(batch)
        logits, target, _ = self(x, label, bbox, mask)

        loss = {}

        loss_logit = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        loss['logit'] = loss_logit

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train/loss", loss['logit'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss['logit']


    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val/loss", loss['logit'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss['logit']

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('eos_token')


        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer

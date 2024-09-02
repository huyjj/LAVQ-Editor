import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
# from data import get_data_loader
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding

import model.Losses as Losses
from .classifier import resnet18_encoder_const_input
# from model import update_average
from .Blocks import DiscriminatorBlock, DiscriminatorTop
from .CustomLayers import (EqualizedConv1d, EqualizedLinear, NoiseLayer1d, BlurLayer1d,
                                 PixelNormLayer, Truncation, LayerEpilogue1d)



class InputBlock(nn.Module):
    """
    The first block (4x4 "pixels") doesn't have an input.
    The result of the first convolution is just replaced by a (trained) constant.
    We call it the InputBlock, the others GSynthesisBlock.
    (It might be nicer to do this the other way round,
    i.e. have the LayerEpilogue be the Layer and call the conv from that.)
    """

    def __init__(self, nf, dlatent_size, const_input_layer, gain, const_input_dim,
                 use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_diseases, 
                 activation_layer, use_w3=True):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if const_input_dim > 0:
            self.encoder = resnet18_encoder_const_input(out_dim=const_input_dim)
            self.lin = nn.Linear(1, 4)
        self.const_input_dim = const_input_dim
 
        if self.const_input_layer:
                # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = EqualizedLinear(dlatent_size, nf * 16, gain=gain / 4,
                                            use_wscale=use_wscale)
                # tweak gain to match the official implementation of Progressing GAN

        self.epi1 = LayerEpilogue1d(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_diseases, activation_layer, use_w3=use_w3)
        self.conv = EqualizedConv1d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue1d(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_diseases, activation_layer, use_w3=use_w3)

    def forward(self, dlatents_in_range, const_input=None):
        batch_size = dlatents_in_range.size(0)
        if const_input is None:
            if self.const_input_layer:
                x = self.const.expand(batch_size, -1, -1)
                x = x + self.bias.view(1, -1, 1)
            else:
                x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4)
        else:
            x = const_input
            
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])

        return x


class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain,
                 use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_diseases, 
                 activation_layer, use_w3=True):
        super().__init__()

        if blur_filter:
            blur = BlurLayer1d(blur_filter)
        else:
            blur = None

        self.conv0_up = EqualizedConv1d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                        intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue1d(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_diseases, activation_layer, use_w3=use_w3)
        self.conv1 = EqualizedConv1d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue1d(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_diseases, activation_layer, use_w3=use_w3)

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x



class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.
        # TODO

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', PixelNormLayer()))

        # Mapping layers. (apply_bias?)
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)  # (b, lead, dim)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)  # 纬度加了一个-1
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=12, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=256, const_input_dim=512, use_w3=True,
                 use_diseases=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu',
                 use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=None,
                 structure='linear', **kwargs):
        '''

        '''
        super().__init__()

        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 #- 1

        self.num_layers = resolution_log2 * 2 #- 2
        self.num_diseases = self.num_layers if use_diseases else 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # Early layers.
        self.init_block = InputBlock(nf(1), dlatent_size, const_input_layer, gain, const_input_dim, use_wscale,
                                     use_noise, use_pixel_norm, use_instance_norm, use_diseases, act, use_w3)
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv1d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_diseases, act, use_w3))
            rgb_converters.append(EqualizedConv1d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None, const_input=None):
        """
            forward pass of the Generator
            :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
            :param labels_in:
            :param depth: current depth from where output is required  选择第几层的output
            :param alpha: value of alpha for fade-in effect
            :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            x = self.init_block(dlatents_in[:, 0:2], const_input=const_input)
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':
            x = self.init_block(dlatents_in[:, 0:2], const_input=const_input)

            if depth > 0:
                for i, block in enumerate(self.blocks[:depth - 1]):
                    x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](self.blocks[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return images_out



class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512, const_input_dim=512, # const_input_dim for encoder samples
                 conditional=False, n_classes=0, truncation_psi=0.7, fmap_max=512, 
                 truncation_cutoff=8, dlatent_avg_beta=0.995,
                 disease_mixing_prob=0.9, use_w3=False, **kwargs):
        
        super(Generator, self).__init__()

        if conditional:
            assert n_classes > 0, "Conditional generation requires n_class > 0"
            self.class_embedding = nn.Embedding(n_classes, latent_size)
            latent_size *= 2

        self.conditional = conditional
        self.disease_mixing_prob = disease_mixing_prob
        self.latent_size = latent_size
    
        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, dlatent_size=dlatent_size, fmap_max=fmap_max,
                                      const_input_dim=const_input_dim, use_w3=use_w3, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in, depth, alpha, labels_in=None, const_input=None):
        if not self.conditional:
            if labels_in is not None:
                warnings.warn(
                    "Generator is unconditional, labels_in will be ignored")
        else:
            assert labels_in is not None, "Conditional discriminatin requires labels"
            embedding = self.class_embedding(labels_in)
            latents_in = torch.cat([latents_in, embedding], 1)

        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform disease mixing regularization.
            if self.disease_mixing_prob is not None and self.disease_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
                    latents_in.device)
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.disease_mixing_prob else cur_layers
                dlatents_in = torch.where(layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha, labels_in=labels_in, const_input=const_input)
        # print(fake_images.shape)
        return fake_images 

class Discriminator(nn.Module):

    def __init__(self, resolution, num_channels=3, conditional=False, disease_model=None,
                 n_classes=6, fmap_base=8192, fmap_decay=1.0, fmap_max=512, 
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4,
                 mbstd_num_features=1, blur_filter=None, structure='linear',
                 **kwargs):
        super(Discriminator, self).__init__()

        if conditional:
            assert n_classes > 0, "Conditional Discriminator requires n_class > 0"
            num_channels *= 2
            self.embeddings = []

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.conditional = conditional
        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure


        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, use_wscale=use_wscale, activation_layer=act,
                                             blur_kernel=blur_filter))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv1d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))
            # Create embeddings for various inputs:
            if conditional:
                r = 2 ** (res)
                self.embeddings.append(
                    Embedding(n_classes, (num_channels // 2) * r * r))

        if self.conditional:
            self.embeddings.append(nn.Embedding(
                n_classes, (num_channels // 2) * 4 * 4))
            self.embeddings = nn.ModuleList(self.embeddings)

        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv1d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)
        if disease_model is None:
            self.final_classifier = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                                in_channels=nf(2), intermediate_channels=nf(2), output_features=n_classes,
                                                gain=gain, use_wscale=use_wscale, activation_layer=act)
        else:
            self.final_classifier = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                                in_channels=nf(2), intermediate_channels=nf(2), output_features=fmap_max,
                                                gain=gain, use_wscale=use_wscale, activation_layer=act)
        
        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool1d(2)

    def forward(self, images_in, depth, alpha=1., labels_in=None):
        
        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.conditional:
            assert labels_in is not None, "Conditional Discriminator requires labels"

        if self.structure == 'fixed':
            if self.conditional:
                embedding_in = self.embeddings[0](labels_in)
                embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                 images_in.shape[2],
                                                 images_in.shape[3])
                images_in = torch.cat([images_in, embedding_in], dim=1)
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)
            
        elif self.structure == 'linear':
            if depth > 0:
                if self.conditional:
                    embedding_in = self.embeddings[self.depth -
                                                   depth - 1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)
                
                residual = self.from_rgb[self.depth -
                                         depth](self.temporaryDownsampler(images_in))

                straight = self.blocks[self.depth - depth -
                                       1](self.from_rgb[self.depth - depth - 1](images_in))
                x = (alpha * straight) + ((1 - alpha) * residual)
 
                for block in self.blocks[(self.depth - depth):]:
                    x = block(x)
            else:
                if self.conditional:
                    embedding_in = self.embeddings[-1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)
                x = self.from_rgb[-1](images_in)
            # print(x.shape)
            scores_out = self.final_block(x)
            logit_out = self.final_classifier(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out, logit_out


class Ecgedit:

    def __init__(self, disease_model, structure, resolution, num_channels, latent_size, dlatent_size=4096, const_input_dim=512,
                 g_args=None, d_args=None, g_opt_args=None, d_opt_args=None, conditional=False, logger=None, fmap_max=512,
                 n_classes=0, loss="relativistic-hinge", drift=0.001, d_repeats=1, lr=0.0003,
                 use_ema=False, ema_decay=0.999, device=torch.device("cuda")):


        # state of the object
        assert structure in ['fixed', 'linear']

        # Check conditional validity
        if conditional:
            assert n_classes > 0, "Conditional GANs require n_classes > 0"
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        self.latent_size = latent_size
        self.device = device
        self.d_repeats = d_repeats
        self.conditional = conditional
        self.n_classes = n_classes

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.logger = logger
        self.lr = lr
        self.disease_model = disease_model.to(self.device)
        # Create the Generator and the Discriminator


        self.gen = Generator(num_channels=num_channels,
                                resolution=resolution,
                                latent_size=latent_size//2,
                                dlatent_size=dlatent_size//2,
                                structure=self.structure,
                                conditional=self.conditional,
                                n_classes=self.n_classes,
                                style_mixing_prob=0,
                                const_input_dim=const_input_dim,
                                fmap_max=fmap_max,
                                #  **g_args
                                ).to(self.device)

        self.dis = Discriminator(num_channels=num_channels,
                                 resolution=resolution,
                                 structure=self.structure,
                                 conditional=self.conditional,
                                 n_classes=self.n_classes,
                                 disease_model=disease_model,
                                 fmap_max=fmap_max,
                                #  **d_args
                                 ).to(self.device)

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(learning_rate=self.lr, beta_1=0, beta_2=0.99, eps=1e-8)
        self.__setup_dis_optim(learning_rate=self.lr, beta_1=0, beta_2=0.99, eps=1e-8)

        # define the loss function used for training the GAN
        self.drift = drift
        self.loss = self.__setup_loss(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        self.gen_optim = torch.optim.Adam([{"params": self.gen.parameters()}, {"params": self.disease_model.parameters()}], lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_loss(self, loss):
        loss_func = Losses.EditorGAN(self.dis, self.disease_model)
        return loss_func


    def save_ckpt(self, epoch, val_loss, val_f1, val_acc, stage):
        state = {"state_dict": self.gen.state_dict(), "epoch": epoch, "loss": val_loss, 
                 'f1': val_f1, 'acc': val_acc, 'lr': self.lr, 'stage': stage, "optimizer": self.gen_optim.state_dict(),
                }
        d_state = {"state_dict": self.dis.state_dict(), "epoch": epoch, "loss": val_loss, "optimizer": self.dis_optim.state_dict(),
                 'f1': val_f1, 'acc': val_acc, 'lr': self.lr, 'stage': stage}
        f_state = {"state_dict": self.disease_model.state_dict(), "epoch": epoch, "loss": val_loss, 
                 'f1': val_f1, 'acc': val_acc, 'lr': self.lr, 'stage': stage}
        
        return state, d_state, f_state


    def optimize_generator(self, disease_batch, disease_descri, real_batch, real_descri, depth, alpha, const_input=None):
        real_samples = real_batch[:, :12]
        real_pos = real_batch[:, 12:]
        disease_samples = disease_batch[:, :12]
        disease_pos = disease_batch[:, 12:]

        disease, const_disease, q_loss_s, prefix_s, _ = self.disease_model(disease_samples, disease_pos, disease_descri)
        disease_r, const_r, q_loss_r, prefix_r, _ = self.disease_model(real_samples, real_pos, real_descri)

        edited_samples = self.gen(disease, depth, alpha, None, const_r)
        recon_samples = self.gen(disease_r, depth, alpha, None, const_r)

        fake_disease, const_f, q_loss_f, prefix_f, _ = self.disease_model(edited_samples, real_pos, disease_descri)
        
        
        loss, f_loss, rec_loss, sim_loss = self.loss.gen_loss(real_samples, edited_samples, recon_samples, 
                                                              disease, fake_disease.detach(), depth, alpha)
        loss = loss + q_loss_r #+ q_loss_s
        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=10.)
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item(), f_loss.item(), rec_loss.item(), sim_loss.item(), q_loss_r.item(), prefix_r

    def optimize_discriminator(self, disease_batch, disease_descri, real_batch, real_descri, depth, alpha, const_input=None):
        
        real_samples = real_batch[:, :12]
        real_pos = real_batch[:, 12:]
        disease_samples = disease_batch[:, :12]
        disease_pos = disease_batch[:, 12:]
        disease, const_disease, q_loss_s, prefix_s, _ = self.disease_model(disease_samples, disease_pos, disease_descri)
        disease_r, const_r, q_loss_r, prefix_r,_  = self.disease_model(real_samples, real_pos, real_descri)
        disease = disease.detach()
        disease_r = disease_r.detach()
        const_r = const_r.detach()

        loss_val = 0
        loss_cls = 0
        loss_ori = 0
        for _ in range(self.d_repeats):
            # generate a batch of samples
            fake_samples = self.gen(disease, depth, alpha, None, const_r).detach()
            loss, cls_loss, ori_loss = self.loss.dis_loss(
                    real_samples, fake_samples, disease_r, depth, alpha)
            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()
            loss_cls += cls_loss.item()
            loss_ori += ori_loss.item()

        return loss_val / self.d_repeats, loss_val / self.d_repeats, loss_ori / self.d_repeats

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        from torch.nn.functional import interpolate
        from torchvision.utils import save_image

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True, pad_value=128, padding=1)




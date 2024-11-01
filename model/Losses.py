"""
Here is the original copy right of the code:
-------------------------------------------------
   File Name:    Losses.py
   Author:       Zhonghao Huang
   Date:         2019/10/21
   Description:  Module implementing various loss functions
                 Copy from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.criterion = BCEWithLogitsLoss()
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha, labels_in=labels)
        f_preds = self.dis(fake_samps, height, alpha, labels_in=labels)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        preds = self.dis(fake_samps, height, alpha, labels_in=labels)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = (torch.mean(nn.ReLU()(1 - r_preds)) +
                torch.mean(nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        return -torch.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))


class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img, height, alpha):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img, height, alpha)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty
    
    def R1Penalty_RC(self, real_img, height, alpha):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit, _ = self.dis(real_img, height, alpha)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        f_preds = self.dis(fake_samps, height, alpha)

        return torch.mean(nn.Softplus()(-f_preds))
    
    def gen_loss_RC(self, real_samps, fake_samps, recon_sample, label_t, height, alpha):
        f_preds, f_logit = self.dis(fake_samps, height, alpha)
        f_loss = torch.mean(nn.Softplus()(-f_preds))
        
        rec_loss = torch.sum(torch.abs(recon_sample - real_samps)) / recon_sample.numel()
        cls_loss = Focal_loss(f_logit, label_t)
        return  f_loss + rec_loss + cls_loss, f_loss, rec_loss, cls_loss
    
    def dis_loss_RC(self, real_samps, fake_samps, label_r, height, alpha, r1_gamma=10.0):
        # Obtain predictions
        '''
        logit_real: classifier for real_samples(have)
        '''
        r_preds, r_logit = self.dis(real_samps, height, alpha)
        f_preds, f_logit = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty_RC(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        cls_loss = Focal_loss(r_logit, label_r)
        return loss + cls_loss, cls_loss, loss


class EditorGAN(GANLoss):
    def __init__(self, dis, style_model):
        super().__init__(dis)
        self.style_model = style_model
    
    def R1Penalty(self, real_img, height, alpha):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit, _ = self.dis(real_img, height, alpha)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    
    def gen_loss(self, real_samps, fake_samps, recon_sample, style, fake_style, height, alpha, tau=1):
        f_preds, _ = self.dis(fake_samps, height, alpha)
        f_loss = torch.mean(nn.Softplus()(-f_preds))
        
        rec_loss = torch.sum(torch.abs(recon_sample - real_samps)) / recon_sample.numel()
        # 计算与相似度：style 与fake style?
        sim_loss = 1 - (F.cosine_similarity(fake_style, style, dim=-1) / tau).mean()
    
        return  f_loss + rec_loss + sim_loss, f_loss, rec_loss, sim_loss

    def gen_loss_wo_sim(self, real_samps, fake_samps, recon_sample, style, fake_style, height, alpha, tau=1):
        f_preds, _ = self.dis(fake_samps, height, alpha)
        f_loss = torch.mean(nn.Softplus()(-f_preds))
        
        rec_loss = torch.sum(torch.abs(recon_sample - real_samps)) / recon_sample.numel()
        # 计算与相似度：style 与fake style?
        # sim_loss = 1 - (F.cosine_similarity(fake_style, style, dim=-1) / tau).mean()
    
        return  f_loss + rec_loss, f_loss, rec_loss, rec_loss
    

    def gen_loss_wo_sim_wo_rec(self, real_samps, fake_samps, recon_sample, style, fake_style, height, alpha, tau=1):
        f_preds, _ = self.dis(fake_samps, height, alpha)
        f_loss = torch.mean(nn.Softplus()(-f_preds))
        
        # rec_loss = torch.sum(torch.abs(recon_sample - real_samps)) / recon_sample.numel()
        # 计算与相似度：style 与fake style?
        # sim_loss = 1 - (F.cosine_similarity(fake_style, style, dim=-1) / tau).mean()
    
        return  f_loss , f_loss, f_loss, f_loss
    
    def gen_loss_wo_rec(self, real_samps, fake_samps, recon_sample, style, fake_style, height, alpha, tau=1):
        f_preds, _ = self.dis(fake_samps, height, alpha)
        f_loss = torch.mean(nn.Softplus()(-f_preds))
        
        # rec_loss = torch.sum(torch.abs(recon_sample - real_samps)) / recon_sample.numel()
        # 计算与相似度：style 与fake style?
        sim_loss = 1 - (F.cosine_similarity(fake_style, style, dim=-1) / tau).mean()
    
        return  f_loss + sim_loss, f_loss, f_loss, sim_loss
        
    def dis_loss_wo_sim(self, real_samps, fake_samps, style, height, alpha, r1_gamma=10.0, tau=1):
        # Obtain predictions
        '''
        logit_real: classifier for real_samples(have)
        '''
        
        r_preds, r_style = self.dis(real_samps, height, alpha)
        f_preds, f_style = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        # sim_loss = 1 - (F.cosine_similarity(r_style, style, dim=-1) / tau).mean()
        
        return loss, loss, loss
    
    def dis_loss(self, real_samps, fake_samps, style, height, alpha, r1_gamma=10.0, tau=1):
        # Obtain predictions
        '''
        logit_real: classifier for real_samples(have)
        '''
        
        r_preds, r_style = self.dis(real_samps, height, alpha)
        f_preds, f_style = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        sim_loss = 1 - (F.cosine_similarity(r_style, style, dim=-1) / tau).mean()
        
        return loss + sim_loss, sim_loss, loss


def Focal_loss(pred, target, gamma=2, reduction='mean'):
    log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, num)
    logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
    logpt = logpt.view(-1)  # 降维，shape=(bs)
    ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
    pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
    focal_loss = (1 - pt) ** gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
    if reduction == "mean":
        return torch.mean(focal_loss)
    if reduction == "sum":
        return torch.sum(focal_loss)
    return focal_loss
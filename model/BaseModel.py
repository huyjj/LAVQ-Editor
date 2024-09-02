import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import enisum
from .classifier import resnet18_encoder, resnet34_encoder
from transformers import BertModel, BertTokenizer
from .Attention import LinearCrossAttention
from .VQ import EMAVectorQuantizer
from scipy.optimize import linear_sum_assignment

class MLP_block(nn.Module):
    '''
    '''
    def __init__(self, output_dim, feature_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        return output





class VQSeparator(nn.Module):
    def __init__(self, embedding_dim=512, context_dim=1024, resolution=4096, language_model='./Bio_ClinaBert'):
        super(VQSeparator, self).__init__()
        self.context_dim = context_dim
        self.ecg_encoder = resnet34_encoder(num_classes=embedding_dim)
        self.pos_encoder = resnet18_encoder(out_dim=embedding_dim//2)
        self.language_model = BertModel.from_pretrained(language_model, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(language_model)
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.LinearCrossAtt_ecg2text = LinearCrossAttention(dim=4, context_dim=1)
        self.LinearCrossAtt_pos2text = LinearCrossAttention(dim=1, context_dim=1)
        self.des_mapper = MLP_block(output_dim=context_dim//4, feature_dim=768)
        self.quant_conv = nn.Linear(embedding_dim, 4)
        self.codebook = EMAVectorQuantizer(latent_dim=4, num_codebook_vectors=embedding_dim, beta=0.25)


    def text_encoder(self, x):
        tokens = self.tokenizer.batch_encode_plus(x, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()
        outputs = self.language_model(input_ids, attention_mask)
        text_representation = outputs.last_hidden_state[:, 0, :]
        text_embed = self.des_mapper(text_representation)
        return text_embed
    
    def forward(self, ecg_data, pos, text_data):
        text_embed = self.text_encoder(text_data) 

        ecg_embed = self.ecg_encoder(ecg_data[:, :12])
        quant_conv_embed = self.quant_conv(ecg_embed)
        mask = (torch.sigmoid(self.LinearCrossAtt_ecg2text(quant_conv_embed, text_embed.unsqueeze(-1))) > 0.5).int()
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_embed, mask)

        ecg_const = quant_conv_embed * (1 - mask) 
        ecg_disease = codebook_mapping * mask  
        ecg_disease = torch.mean(ecg_disease, dim=-1)
        
        return ecg_disease, ecg_const, q_loss, codebook_indices[0], mask


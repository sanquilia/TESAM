import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel, BertConfig
from model.tesam import ResNetLayer, BertLayer, MultiheadCrossAttention, CrossmodalSentimentEncoder, Pooler
global config
config=BertConfig.from_pretrained('./weights/config.json')

class TESAM4align(nn.Module):
    def __init__(self, num_classes, cnn_out_dim=512, bert_out_dim=512,i=1,s=2):
        super(TESAM4align, self).__init__()
        self.cnn=ResNetLayer(cnn_out_dim)
        self.bert=BertLayer(bert_out_dim)

    def forward(self, img, txt):
        return self.forward1(img, txt)
        
    def forward1(self, img, txt):
        input_id, input_mask, segment_id, ocr_id, ocr_mask=txt #,target_id,target_mask
        # with torch.no_grad():
        x1=self.cnn(img)
        x2=self.bert((input_id, input_mask, segment_id))
        
        return x1,x2
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel, BertConfig
from model.resnet import resnet18
global config
from einops import repeat
from einops.layers.torch import Rearrange
config=BertConfig.from_pretrained('./weights/config.json')

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 1, embed_dim: int = 768, img_size: int = 7):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, embed_dim))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, embed_dim))
        
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
    
class ResNetLayer(nn.Module):
    def __init__(self,cnn_out_dim=512,i=1):
        super(ResNetLayer,self).__init__()
        self.backbone=resnet18()
        self.pos_emb=PatchEmbedding(in_channels=512,embed_dim=cnn_out_dim)
        self.backbone.load_state_dict(torch.load('./weights/resnet18_backbone.pt'),strict=False)
        self.img_encoder = ImageEncoder(n_layers=i,cnn_out_dim=cnn_out_dim)

    def forward(self, img):
        x,_=self.backbone(img)
        x=self.pos_emb(x)
        x=self.img_encoder(x)
        return x #(batchsize,49,2048)
    
class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc(x[:, 0])
        out = self.tanh(out)
        return out

class LayerNorm(nn.Module):
        def __init__(self, hidden_size=768, eps=1e-12):
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 2.0**.5))

def swish(x):
    return x * torch.sigmoid(x)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads=12):
        super(MultiheadCrossAttention, self).__init__()
        self.head_dim = n_heads * int(dim / n_heads)
        self.query = nn.Linear(dim, self.head_dim)
        self.key = nn.Linear(dim, self.head_dim)
        self.value = nn.Linear(dim, self.head_dim)
        self.dropout = nn.Dropout(0.01)

    def get_score(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        Q = self.get_score(self.query(x))
        K = self.get_score(self.key(x))
        V = self.get_score(self.value(x))

        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores / self.head_size ** .5

        if mask is not None:
            scores = scores * mask.unsqueeze(1).unsqueeze(2)

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        output = torch.matmul(probs, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*output_shape)
        return output

class FCN(nn.Module):
    def __init__(self, config):
        super(FCN, self).__init__()
        self.fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = gelu

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        return x
     
class AddNorm(nn.Module):
    def __init__(self, config):
        super(AddNorm, self).__init__()
        self.fc = nn.Linear(3072, 768)
        self.ln = LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x, x0):
        x = self.fc(x)
        x = self.dropout(x)
        x = self.ln(x + x0)
        return x

class BertLayer(nn.Module):
    def __init__(self,bert_out_dim=512):
        super(BertLayer,self).__init__()
        self.model = BertModel.from_pretrained('./weights/pytorch_model.bin', config = BertConfig.from_pretrained('./weights/config.json')) #.from_pretrained('./weights', config = model_config)

    def forward(self, text):
        input_id, input_mask, segment_id = text
        x = self.model(input_id, input_mask, segment_id)[0]
        return x

class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=12):
        super(MultiheadCrossAttention, self).__init__()
        self.head_dim = n_heads * int(dim / n_heads)
        self.query = nn.Linear(dim, self.head_dim)
        self.key = nn.Linear(dim, self.head_dim)
        self.value = nn.Linear(dim, self.head_dim)
        self.dropout = nn.Dropout(0.01)

    def get_score(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, key_mask=None):

        Q = self.get_score(self.query(query))
        K = self.get_score(self.key(key))
        V = self.get_score(self.value(key))

        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores / self.head_size ** .5

        if key_mask is not None:
            scores = scores * key_mask.unsqueeze(1).unsqueeze(2)

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        output = torch.matmul(probs, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.head_dim,)
        output = output.view(*output_shape)
        return output

class ImageEncoderLayer(nn.Module):
    def __init__(self,cnn_out_dim=768,sa=True):
        super(ImageEncoderLayer,self).__init__()
        i=cnn_out_dim
        self.sa=sa
        if self.sa:
            self.attention=MultiheadSelfAttention(config)
        self.intermediate = FCN(config)
        self.output = AddNorm(config)
        self.norm_img=LayerNorm(i)
    def forward(self, img):
        if self.sa:
            att_out=self.norm_img(self.attention(img)+img)
            x=self.intermediate(att_out)
            x=self.output(x, att_out)
        else:
            x=self.intermediate(img)
            x=self.output(x, img)
            # x=self.norm_img(x+img)            
        return x #(b,50,512)
    
class ImageEncoder(nn.Module):
    def __init__(self,n_layers=1,cnn_out_dim=768,sa=True):
        super(ImageEncoder,self).__init__()
        self.layers=nn.Sequential(*[ImageEncoderLayer(cnn_out_dim,sa) for i in range(n_layers)])
    def forward(self, img):
        x=self.layers(img)        
        return x #(b,50,512)
    

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.fc = FCN(config)
        self.addnorm = AddNorm(config)
    def forward(self, x):
        x0 = self.fc(x)
        out = self.addnorm(x0, x)
        return out
    
class CrossmodalSentimentEncoder(nn.Module):
    def __init__(self, n_layers=1, cnn_out_dim=768, bert_out_dim=768):
        super(CrossmodalSentimentEncoder, self).__init__()
        i=cnn_out_dim
        self.att_img=MultiheadCrossAttention(cnn_out_dim)
        self.att_text=MultiheadCrossAttention(cnn_out_dim)
        self.norm_img=LayerNorm(cnn_out_dim)
        self.norm_text=LayerNorm(bert_out_dim)
        # self.img_encoder = nn.ModuleList([Transformer(config) for i in range(n_layers)])
        self.img_encoder = ImageEncoder(n_layers,cnn_out_dim,sa=False)
        self.text_encoder=TextEncoder(config)
        self.norm_img_=LayerNorm(cnn_out_dim)
        self.norm_text_=LayerNorm(bert_out_dim)


    def forward(self, input,input_mask):
        img,txt=input
        # input_mask=(1.0 - input_mask) * -1000000.0
        x1=self.att_img(txt,img)
        x2=self.att_text(img,txt,input_mask)

        x1_att=self.norm_img(x1+img)
        x2_att=self.norm_text(x2+txt)

        x1_=self.img_encoder(x1_att)
        x2_=self.text_encoder(x2_att)

        x1_out=self.norm_img_(x1_+x1_att)
        x2_out=self.norm_text_(x2_+x2_att)

        return (x1_out,x2_out)
    
    
class TESAM(nn.Module):
    def __init__(self, num_classes, cnn_out_dim=512, bert_out_dim=512,i=1,s=2):
        super(TESAM, self).__init__()
        self.cnn=ResNetLayer(cnn_out_dim)
        self.cnn.load_state_dict(torch.load('./weights/cnn.pt'))
        self.bert=BertLayer(bert_out_dim)
        self.bert.load_state_dict(torch.load('./weights/bert.pt'))
        self.att=MultiheadCrossAttention(cnn_out_dim)

        self.CSEstack=nn.ModuleList([CrossmodalSentimentEncoder(1,cnn_out_dim,bert_out_dim) for i in range(s)])
        self.txt_pooler=Pooler(config)
        self.txt_ds = nn.Linear(in_features=bert_out_dim, out_features=num_classes,bias=True)
        self.img_pooler=Pooler(config)
        self.img_ds=nn.Linear(in_features=bert_out_dim, out_features=num_classes,bias=True)
        self.w=nn.Parameter(torch.ones(1,num_classes,requires_grad=True)*.5)

    def forward(self, img, txt):
        return self.forward1(img, txt)
        
    def forward1(self, img, txt):
        input_id, input_mask, segment_id, ocr_id, ocr_mask=txt #,target_id,target_mask
        # with torch.no_grad():
        x1=self.cnn(img)
        x2=self.bert((input_id, input_mask, segment_id))
        x3=self.bert((ocr_id, ocr_mask, segment_id))
        x1_att=self.att(x3,x1,input_mask)
        x1=f.normalize(x1+x1_att)
        
        for layer in self.CSEstack:
            (x1,x2)=layer((x1,x2),input_mask)
        x1=self.img_ds(self.img_pooler(x1))
        x2=self.txt_ds(self.txt_pooler(x2))
        x=self.w*x1+(1-self.w)*x2
        
        return x
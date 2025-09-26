import torch.nn as nn
import torch
# 2) 1D-CNN MODEL
class CNN1DBinaryClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels,  32, kernel_size=7, padding=3, stride=1), # 7=>35 min (288)
            nn.ReLU(),
            nn.Conv1d(32,  64, kernel_size=7, padding=3, stride=2), # 14*5=80 min (144)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3, stride=2), #28*5 = 160 min (72)
            nn.ReLU(),
            nn.Conv1d(128,256, kernel_size=7, padding=3,stride=2), #56*5 = 320 min (36)
            nn.ReLU(), # (B,256,250)
            nn.Conv1d(256,512,kernel_size =3, padding=1, stride=2),#80*5 = 400 min (18)
            nn.ReLU(), # (B,512,125)
            nn.Conv1d(512,512,kernel_size =3, padding=1, stride=2),#128*5 = 640 min (10 hrs) (9)
            nn.ReLU(), # (B,512,62)
            nn.Conv1d(512,1024,kernel_size =3, padding=1, stride=2),#224*5 = 1240 min  (20 hrs) (5)
            nn.ReLU(), # (B,1024,31)
            nn.AdaptiveAvgPool1d(1),     # → (B,512,1)
            nn.Flatten(),                # → (B,512)
            #nn.Linear(1024, 512),
            nn.ReLU(), # (B,1024,31)
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.network(x)

from transformers import AutoModel
import torch.nn as nn

class TransformerFusion(nn.Module):
    def __init__(self, model, **kwargs):
        super(TransformerFusion, self).__init__()
        
        self.model = model
        self.emblist = nn.ModuleList()
        self.backbone = nn.Sequential(
            *list(model.network.children())[:-3]
            )
        print(self.backbone)
        self.outputofmodeldim = 1024
        self.embed_dim=256
        encoder_layer = nn.TransformerEncoderLayer(
            d_model= self.embed_dim,
            nhead=8,
            dim_feedforward=4*self.embed_dim,
            dropout=0.1,
            batch_first=True       # so input is (B, seq_len, embed_dim)
        )
        # stack 4 of them
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=8
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        self.linear = nn.Linear(self.outputofmodeldim,self.embed_dim)
        self.output_linear = nn.Linear(self.embed_dim,2)
        
    def forward(self, glucose_vals): #glucose_vals is 2016 glucose values (B, 2016)
        chunk_embed_list = [] 

        for offset in range(7):
            chunk = glucose_vals[:, :, offset*288:(offset+1)*288]
            chunk_embed = self.backbone(chunk)
            chunk_embed = chunk_embed.squeeze()
            B,dim = chunk_embed.size()
          
            chunk_embed = self.linear(chunk_embed.reshape((B,1,dim)))
            chunk_embed_list.append(chunk_embed)
        

        chunk_embeddings = torch.cat(chunk_embed_list, dim=1) 
        chunk_embeddings = chunk_embeddings.reshape((B,len(chunk_embed_list), self.embed_dim)) 
    
        cls_tokens = self.cls_token.expand(B, -1, -1)
        concat_data = torch.cat((cls_tokens, chunk_embeddings), dim=1)

        x = self.transformer_encoder(concat_data)
        cls_after_transformer = x[:,0,:]
        #print("cls after t size", cls_after_transformer.size())
        x = self.output_linear(cls_after_transformer)
        return x
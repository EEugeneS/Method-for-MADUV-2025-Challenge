import torch, torch.nn as nn, torch.nn.functional as F
import sys
from beats import BEATs

from pooling import AttentivePooling, AvgPool
import torchaudio.transforms as T

class beats_model(nn.Module):

    def __init__(self,  
                 chkpt_path="./beats/BEATs_iter3plus_AS2M.pt"):
        super().__init__()
        
        checkpoint = torch.load(chkpt_path)
        cfg = BEATs.BEATsConfig(checkpoint['cfg'])
        self.model = BEATs.BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
        
        self.predictor = nn.Linear(cfg.encoder_embed_dim, 2)
        self.pool = AttentivePooling(2)
        
        # self.pool2 = AvgPool()
        # self.predictor2 = nn.Linear(24, 2)
        
        # Spectrogram Augmentation
        self.specAug = nn.Sequential(
            T.FrequencyMasking(15),
            T.TimeMasking(10),
        )
        
        
    @torch.no_grad()
    def infer(self, x):
        self.model.eval()
        
        x = self.model.extract_features(x, padding_mask=torch.zeros_like(x).bool())[0]
        x = self.predictor_dropout(x)
        x = self.predictor(x)
        x = self.pool(x)
        # x = self.pool2(x)
        # x = self.predictor2(x)
        
        
        return x
        

    def forward(self, x, padding_mask=None):
        x = torch.vstack([self.specAug(fb[None]) for fb in x])
        
        if padding_mask is None:
            padding_mask = torch.zeros_like(x).bool()
        x = self.model.extract_features(x, padding_mask=padding_mask)[0]
        x = self.predictor_dropout(x)
        x = self.predictor(x)
        x = self.pool(x)
        # x = self.pool2(x)
        # x = self.predictor2(x)
        
        return x

if __name__ == "__main__":

    model = beats_model()
    audio_input_16khz = torch.randn(2, 61, 128)
    mid_emb = model.forward(audio_input_16khz)
    print(mid_emb.shape)



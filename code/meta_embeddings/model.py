from torch import nn
import torch
import torch.nn.functional as f
import numpy as np
#from sklearn.preprocessing import normalize

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, distance_type='mse'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = None
        if distance_type == 'mse':
            self.distance = lambda anchor, targets: (anchor - targets).pow(2).sum(1)
        elif distance_type == 'cosine':
            self.distance = lambda anchor, targets: 1.0 - nn.CosineSimilarity()(anchor, targets)

    def forward(self, anchor, positive, negative, size_average=True):
        #distance_positive = (anchor - positive).pow(2).sum(1)#.pow(.5)
        #distance_negative = (anchor - negative).pow(2).sum(1)#.pow(.5)
        
        distance_positive = self.distance(anchor, positive)
        distance_negative = self.distance(anchor, negative)

        #distance_positive = 1.0 - nn.CosineSimilarity()(anchor, positive)
        #distance_negative = 1.0 - nn.CosineSimilarity()(anchor, negative)

        losses = f.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class AEME(nn.Module):
    def __init__(self, margin=0.1, distance_type='mse'):
        super(AEME, self).__init__()
        self.margin = margin
        self.distance_type = distance_type

    def calc_triplet_loss(self, meta, pos_constrains, neg_constrains):
        loss = 0.0
        loss_obj = TripletLoss(margin=self.margin, distance_type=self.distance_type)
        for i in range(pos_constrains.shape[1]):
            loss += loss_obj(meta, pos_constrains[:, i, :], neg_constrains[:, i, :], size_average=True)
        loss /= pos_constrains.shape[1]
        return loss

    def calc_loss(self, inputs, outputs, wv_weights):
        loss = 0.0
        bs = inputs[0].shape[0]
        for i in range(len(inputs)):
            #mse_loss = nn.MSELoss()(inputs[i], outputs[i])
            cos_loss = nn.CosineEmbeddingLoss()(inputs[i], outputs[i], target=torch.ones(bs, device=torch.device('cuda')))
            #kl_loss = -nn.KLDivLoss(reduction='batchmean')(f.normalize(outputs[i], dim=-1, p=2), inputs[i])
            loss += wv_weights[i] * cos_loss#(kl_loss + cos_loss)#(mse_loss + cos_loss + kl_loss)
        return loss / len(inputs)

class CAEME(AEME):
    def __init__(self, src_emb_shapes, wv_weights=None, alpha=0.5, margin=0.1, distance_type='mse'):
        super(CAEME, self).__init__(margin, distance_type)
        if wv_weights is None:
            wv_weights = np.ones(len(src_emb_shapes), dtype=np.float32)
        wv_weights = wv_weights / np.linalg.norm(wv_weights)
        print(f'weights = {wv_weights}')
        self.wv_weights = torch.tensor(wv_weights, dtype=torch.float32).to('cuda')

        self.alpha = alpha

        self.encoders = nn.ModuleList()
        for src_shape in src_emb_shapes:
            self.encoders.append(nn.Sequential(
                #nn.Dropout(0.05),
                nn.Linear(src_shape, src_shape),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))

        self.decoders = nn.ModuleList()
        for src_shape in src_emb_shapes:
            self.decoders.append(nn.Sequential(
                #nn.Dropout(0.1),
                nn.Linear(sum(src_emb_shapes), src_shape)
            ))

    def encode(self, inputs):
        encoded = []
        for i, src_input in enumerate(inputs):
            encoded_out = self.encoders[i](inputs[i])
            encoded.append(encoded_out)

        encoded_meta = f.normalize(torch.cat(encoded, 1), dim=-1, p=2)
        return encoded_meta

    def forward(self, inputs, pos_constrains=None, neg_constrains=None):
        outputs = []
        encoded_meta = self.encode(inputs)
        for i, src_input in enumerate(inputs):
            out = self.decoders[i](encoded_meta)
            outputs.append(out)

        loss = self.calc_loss(inputs, outputs, self.wv_weights)

        if pos_constrains is not None and neg_constrains is not None:
            pos_constrains_reshaped = [constrain_emb.view(-1, constrain_emb.shape[-1]) for constrain_emb in pos_constrains]
            pos_encoded_constrains_meta = self.encode(pos_constrains_reshaped)
            pos_encoded_constrains_meta = pos_encoded_constrains_meta.view(encoded_meta.shape[0], -1, encoded_meta.shape[1])

            neg_constrains_reshaped = [constrain_emb.view(-1, constrain_emb.shape[-1]) for constrain_emb in neg_constrains]
            neg_encoded_constrains_meta = self.encode(neg_constrains_reshaped)
            neg_encoded_constrains_meta = neg_encoded_constrains_meta.view(encoded_meta.shape[0], -1, encoded_meta.shape[1])

            loss_constrains = self.calc_triplet_loss(encoded_meta, pos_encoded_constrains_meta, neg_encoded_constrains_meta)

            return loss + loss_constrains * self.alpha, loss, loss_constrains

        return loss

    def extract(self, inputs):
        return self.encode(inputs)

class AAEME(AEME):
    def __init__(self, src_emb_shapes, aaeme_dim, wv_weights=None, alpha=0.5, margin=0.1, distance_type='mse'):
        super(AAEME, self).__init__(margin, distance_type)
        if wv_weights is None:
            wv_weights = np.ones(len(src_emb_shapes), dtype=np.float32)
        wv_weights = wv_weights / np.linalg.norm(wv_weights)
        print(f'weights = {wv_weights}')
        self.wv_weights = torch.tensor(wv_weights, dtype=torch.float32).to('cuda')

        self.alpha = alpha

        self.encoders = nn.ModuleList()
        for src_shape in src_emb_shapes:
            self.encoders.append(nn.Sequential(
                #nn.Dropout(0.05),
                nn.Linear(src_shape, aaeme_dim),
                nn.ReLU()
            ))

        self.decoders = nn.ModuleList()
        for src_shape in src_emb_shapes:
            self.decoders.append(nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(aaeme_dim, src_shape)
            ))

    def encode(self, inputs):
        encoded = []
        for i, src_input in enumerate(inputs):
            encoded_out = self.encoders[i](inputs[i])
            encoded.append(encoded_out)

        encoded_meta = torch.mean(torch.stack(encoded), 0)
        encoded_meta = f.normalize(encoded_meta, dim=-1, p=2)
        return encoded_meta

    def forward(self, inputs, pos_constrains=None, neg_constrains=None):
        outputs = []
        encoded_meta = self.encode(inputs)
        for i, src_input in enumerate(inputs):
            out = self.decoders[i](encoded_meta)
            outputs.append(out)

        loss = self.calc_loss(inputs, outputs, self.wv_weights)

        if pos_constrains is not None and neg_constrains is not None:
            pos_constrains_reshaped = [constrain_emb.view(-1, constrain_emb.shape[-1]) for constrain_emb in pos_constrains]
            pos_encoded_constrains_meta = self.encode(pos_constrains_reshaped)
            pos_encoded_constrains_meta = pos_encoded_constrains_meta.view(encoded_meta.shape[0], -1, encoded_meta.shape[1])

            neg_constrains_reshaped = [constrain_emb.view(-1, constrain_emb.shape[-1]) for constrain_emb in neg_constrains]
            neg_encoded_constrains_meta = self.encode(neg_constrains_reshaped)
            neg_encoded_constrains_meta = neg_encoded_constrains_meta.view(encoded_meta.shape[0], -1, encoded_meta.shape[1])

            loss_constrains = self.calc_triplet_loss(encoded_meta, pos_encoded_constrains_meta, neg_encoded_constrains_meta)

            return loss + loss_constrains  * self.alpha, loss, loss_constrains
        return loss

    def extract(self, inputs):
        return self.encode(inputs)

class SED(AEME):
    def __init__(self, src_emb_shapes, aaeme_dim, wv_weights=None, alpha=0.5, margin=0.1, distance_type='mse'):
        super(SED, self).__init__(margin, distance_type)
        if wv_weights is None:
            wv_weights = np.ones(len(src_emb_shapes), dtype=np.float32)
        wv_weights = wv_weights / np.linalg.norm(wv_weights)
        print(f'weights = {wv_weights}')
        self.wv_weights = torch.tensor(wv_weights, dtype=torch.float32).to('cuda')

        self.encoder = nn.Sequential(
                #nn.Dropout(0.05),
                nn.Linear(sum(src_emb_shapes), aaeme_dim),
                nn.ReLU()
            )

        self.decoders = nn.ModuleList()
        for src_shape in src_emb_shapes:
            self.decoders.append(nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(aaeme_dim, src_shape)
            ))

    def encode(self, inputs):
        inputs_cat = torch.cat(inputs, 1)
        encoded_meta = self.encoder(inputs_cat)
        encoded_meta = f.normalize(encoded_meta, dim=-1, p=2)
        return encoded_meta

    def forward(self, inputs, constrains=None, targets=None):
        outputs = []
        encoded_meta = self.encode(inputs)
        for i, src_input in enumerate(inputs):
            out = self.decoders[i](encoded_meta)
            outputs.append(out)

        loss = self.calc_loss(inputs, outputs, self.wv_weights)

        if pos_constrains is not None and neg_constrains is not None:
            pos_constrains_reshaped = [constrain_emb.view(-1, constrain_emb.shape[-1]) for constrain_emb in pos_constrains]
            pos_encoded_constrains_meta = self.encode(pos_constrains_reshaped)
            pos_encoded_constrains_meta = pos_encoded_constrains_meta.view(encoded_meta.shape[0], -1, encoded_meta.shape[1])

            neg_constrains_reshaped = [constrain_emb.view(-1, constrain_emb.shape[-1]) for constrain_emb in neg_constrains]
            neg_encoded_constrains_meta = self.encode(neg_constrains_reshaped)
            neg_encoded_constrains_meta = neg_encoded_constrains_meta.view(encoded_meta.shape[0], -1, encoded_meta.shape[1])

            loss_constrains = self.calc_triplet_loss(encoded_meta, pos_encoded_constrains_meta, neg_encoded_constrains_meta)
            
            return loss + loss_constrains  * self.alpha, loss, loss_constrains
        return loss

    def extract(self, inputs):
        return self.encode(inputs)

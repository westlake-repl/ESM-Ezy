import esm
import torch
import torch.nn as nn
class LaccaseModel(nn.Module):
    def __init__(self, pretrained_model_path):
        super(LaccaseModel,self).__init__()
        self.modelEsm, alphabet = esm.pretrained.load_model_and_alphabet_local(pretrained_model_path)
        self.converter = alphabet.get_batch_converter()
        self.dnn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1280, 2)
        )
    
        self._device = None
        
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.modelEsm.parameters()).device
        return self._device

    def forward(self, data):
        out_result = self._get_representations(data)
        out_put = self.dnn(out_result).squeeze()
        return out_put
    
    def _get_layers(self):
        return len(self.modelEsm.layers)
    
    @property
    def layers(self):
        return self.get_layers()
    
    def get_layers(self):
        return self._get_layers()
    
    def get_last_layer_idx(self):
        return self._get_layers()-1
    
    
    def _get_representations(self, data):
        names, sequences, tokens = self.converter(data)
        if self.device is not None:
            tokens = tokens.to(self.device)
        # truncate tokens to max 1022 tokens
        tokens = tokens[:, :1022]
        # get the last layer representations
        last_layer_idx = self._get_layers()
        result = self.modelEsm(tokens, repr_layers=[last_layer_idx])
        out_result = result["representations"][last_layer_idx][:, 0, :].squeeze()
        return out_result
    
    def get_representations(self, data):
        return self._get_representations(data)
    
    def get_names(self, data):
        names, sequences, tokens = self.converter(data)
        return names
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, state_dict_path=None, device=None):
        model = cls(pretrained_model_path)
        if state_dict_path is not None:
            print(f"Loading state dict from {state_dict_path}")
            model.load_state_dict(torch.load(state_dict_path))
        if device is not None:
            model = model.to(device)
            model.device = device
        return model

        



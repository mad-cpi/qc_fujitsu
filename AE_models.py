import torch
import torch.nn as nn



class QuantizerFunc(torch.autograd.Function):
    @staticmethod
    def forward(self, input, npoints=4, dropout=0):
        # self.save_for_backward(input)
        # self.constant = npoints
        if npoints < 0:
            x = torch.sign(input)
            x[x==0] = 1
            return x

        scale = 10**npoints
        input = input * scale
        input = torch.round(input)
        input = input / scale
        return input

    @staticmethod
    def backward(self, grad_output):
        # input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # grad_input[:] = 1
        return grad_input, None
    
class LBAE_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tan = nn.Tanh()
        self.quant = QuantizerFunc.apply
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.LeakyReLU(0.02)
        self.enc0a = torch.nn.Linear(1024, 1024)
        self.enc0b = torch.nn.Linear(1024, 512)

        self.enc1a = torch.nn.Linear(512, 512)
        self.enc1b = torch.nn.Linear(512, 256)

        self.enc2a = torch.nn.Linear(256, 256)
        self.enc2b = torch.nn.Linear(256, 128)

    def forward(self, x):
        x = self.enc0a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.enc0b(x)

        x = self.enc1a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.enc1b(x)

        x = self.enc2a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.enc2b(x)
        x = torch.tanh(x)
        x = self.quant(x)

        return x
class LBAE_decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.LeakyReLU(0.02)
        self.dec0a = torch.nn.Linear(128, 128)
        self.dec0b = torch.nn.Linear(128, 256)

        self.dec1a = torch.nn.Linear(256, 256)
        self.dec1b = torch.nn.Linear(256, 512)

        self.dec2a = torch.nn.Linear(512, 512)
        self.dec2b = torch.nn.Linear(512, 1024)

    def forward(self, x):
        x = self.dec0a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dec0b(x)

        x = self.dec1a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dec1b(x)

        x = self.dec2a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dec2b(x)
        x = torch.sigmoid(x)

        return x

class LBAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LBAE_encoder()
        self.decoder = LBAE_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.tan = nn.Tanh()
        self.quant = QuantizerFunc.apply
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)

        # now for the quantized part
        encoded = torch.tanh(encoded)
        encoded = self.quant(encoded)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode_only(self, x):
        encoded = self.encoder(x)
        encoded = torch.tanh(encoded)
        encoded = self.quant(encoded)
        return encoded
    


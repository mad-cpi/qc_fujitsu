import torch
import torch.nn as nn


class QuantizerFunc(torch.autograd.Function):
    """Quantizer function for binarizing a vector based on sign (+/-)

    """
    @staticmethod
    def forward(self, input):
        """Forward pass quantizes the torch tensor

        Args:
            input (Torch.Tensor): a BxW vector of floats

        Returns:
            Torch.Tensor: a BxW vector of signed integers (-1 or 1)
        """

        x = torch.sign(input)
        x[x==0] = 1
        return x


    @staticmethod
    def backward(self, grad_output):
        """Surrogate backward gradient function that allows the loss derivative to "skip" the non-derivable quantize function

        Args:
            grad_output (_type_): The current backwards gradient

        Returns:
            _type_: the skipped gradient
        """
        # input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # grad_input[:] = 1
        return grad_input, None
    
class LBAE_encoder(torch.nn.Module):
    """Encoder for the Latent Bayes Autoencoder.
        The forward pass codes a 1024 bit vector of (0,1) -> 128 bit vector of (-1, 1)
    """
    def __init__(self):
        super().__init__()
        self.tan = nn.Tanh()
        self.quant = QuantizerFunc.apply

        self.relu = torch.nn.LeakyReLU(0.02)
        self.enc0a = torch.nn.Linear(1024, 1024)
        self.enc0b = torch.nn.Linear(1024, 512)

        self.enc1a = torch.nn.Linear(512, 512)
        self.enc1b = torch.nn.Linear(512, 256)

        self.enc2a = torch.nn.Linear(256, 256)
        self.enc2b = torch.nn.Linear(256, 128)

    def forward(self, x):
        x = self.enc0a(x) + x
        x = self.relu(x)
        x = self.enc0b(x)
        x = self.relu(x)
        x = self.enc1a(x) + x
        x = self.relu(x)
        x = self.enc1b(x)
        x = self.relu(x)
        x = self.enc2a(x) + x
        x = self.relu(x)
        x = self.enc2b(x)

        x = torch.tanh(x)
        x = self.quant(x)

        return x
class LBAE_decoder(torch.nn.Module):
    """Decoder of the LBAE. Takes a 128 bit vector of (-1,1) and produces a 1024 bit vector of (0,1)

    """
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.LeakyReLU(0.02)
        self.dec0a = torch.nn.Linear(128, 128)
        self.dec0b = torch.nn.Linear(128, 256)

        self.dec1a = torch.nn.Linear(256, 256)
        self.dec1b = torch.nn.Linear(256, 512)

        self.dec2a = torch.nn.Linear(512, 512)
        self.dec2b = torch.nn.Linear(512, 1024)

    def forward(self, x):
        x = self.dec0a(x) + x
        x = self.relu(x)
        x = self.dec0b(x)
        x = self.relu(x)
        x = self.dec1a(x) + x
        x = self.relu(x)
        x = self.dec1b(x)
        x = self.relu(x)
        x = self.dec2a(x) + x
        x = self.relu(x)
        x = self.dec2b(x)
        x = torch.sigmoid(x)

        return x

class LBAE(torch.nn.Module):
    """The full Latent Bernoulli Autoencoder. 

    """
    def __init__(self):
        super().__init__()
        self.encoder = LBAE_encoder()
        self.decoder = LBAE_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


import torch
from torch import nn as nn

from models import thops


class ActNorm1d(nn.Module):
    """ActNorm implementation for Point Cloud."""

    def __init__(self, channel: int, dim=1):
        super(ActNorm1d, self).__init__()

        assert dim in [-1, 1, 2]
        self.dim = 2 if dim == -1 else dim
        
        if self.dim == 1:
            size = (1, channel, 1)
            # self.logs = nn.Parameter(torch.zeros(size))
            # self.bias = nn.Parameter(torch.zeros(size))
            self.Ndim = 2
        if self.dim == 2:
            size = (1, 1, channel) 
            # self.logs  = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,device=device,requires_grad=True))
            # self.bias  = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,device=device,requires_grad=True))
            self.Ndim = 1
        
        self.inited = False
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.eps = 1e-6
        # self.register_buffer("_initialized", torch.tensor(False).to(device))
        self.inited = False
    
    def forward(self, x, logdet=None, reverse=False):
        """
            logs is log_std of `mean of channels`
            so we need to multiply by number of voxels
        """
        # if not self._initialized:
        if not self.inited:
            self.__initialize(x)
            # self._initialized = torch.tensor(True)
        
        if not reverse:
            x = (x + self.bias) * torch.exp(self.logs)
            dlogdet = torch.sum(self.logs) * x.shape[self.Ndim]
        else:
            x = x * torch.exp(-self.logs) - self.bias
            dlogdet = -torch.sum(self.logs) * x.shape[self.Ndim]
        
        logdet = logdet + dlogdet
        return x, logdet
    
    def __initialize(self, x):
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        with torch.no_grad():
            dims = [0, 1, 2]
            dims.remove(self.dim)
            bias_ = torch.mean(x.clone(), dim=dims, keepdim=True) * -1.0
            # logs_ = -torch.log(torch.std(x.clone(), dim=dims, keepdim=True) + self.eps)
            vars_ = torch.mean((x.clone() + bias_)**2, dim=dims, keepdim=True)
            logs_ = -torch.log(torch.sqrt(vars_) + self.eps)
            self.logs.data.copy_(logs_.data)
            self.bias.data.copy_(bias_.data)
            self.inited = True


class _ActNorm3d(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3, 4], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3, 4], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = thops.sum(logs) * thops.pixels(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False, offset_mask=None, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not reverse:
            # center and scale
            input = self._center(input, reverse, bias_offset)
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)
        return input, logdet


class ActNorm3d(_ActNorm3d):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 5
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = thops.sum(logs) * thops.pixels(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False, offset_mask=None, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not reverse:
            # center and scale
            input = self._center(input, reverse, bias_offset)
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


# class MaskedActNorm2d(ActNorm2d):
#     def __init__(self, num_features, scale=1.):
#         super().__init__(num_features, scale)

#     def forward(self, input, mask, logdet=None, reverse=False):

#         assert mask.dtype == torch.bool
#         output, logdet_out = super().forward(input, logdet, reverse)

#         input[mask] = output[mask]
#         logdet[mask] = logdet_out[mask]

#         return input, logdet

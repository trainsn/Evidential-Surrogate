import torch
import numpy as np

import pdb

def Gaussian_NLL(y, mu, sigma, reduce=True):
    logprob  = -torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi)) - 0.5 * ((y - mu) / sigma) ** 2
    if reduce:
        # Mean reduction over all dimensions except the batch dimension
        loss = -torch.mean(logprob, dim=tuple(range(1, y.dim())))
        pdb.set_trace()
        sorted_loss, _ = torch.sort(loss)
        trimmed_loss = sorted_loss[3:-3]
        return torch.mean(loss), torch.mean(trimmed_loss)
    else:
        return -logprob

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(torch.tensor(np.pi) / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    
    if reduce:
        # Mean reduction over all dimensions except the batch dimension
        nll = torch.mean(nll, dim=tuple(range(1, y.dim())))
        pdb.set_trace()
        sorted_nll, _ = torch.sort(nll)
        trimmed_nll = sorted_nll[3:-3]
    
    return torch.mean(nll), torch.mean(trimmed_nll) if reduce else nll 

def NIG_Reg(y, gamma, v, alpha, beta, reduce=True):
    # Calculating the absolute error
    error = torch.abs(y - gamma)

    # calculate the regularization based on evidence
    evi = 2 * v + alpha
    reg = error * evi
        
    return torch.mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = torch.chunk(evidential_output, 4, dim=1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)  
    return loss_nll + coeff * loss_reg
import torch
from torch import Tensor


MAX_FLOW = 400 

# def epe_loss(input_flow, target_flow, valid):
def epe_loss(input_flow, target_flow):
    """
    - input_flow: (batch_size, 2, H, W) (means predict flow)
    - target_flow: (batch_size, 2, H, W) (means GT flow)
    - valid: (batch_size, H, W) indicating valid pixels
    """
    # valid = (valid >= 0.5) & (torch.norm(target_flow, p=2, dim=1) < MAX_FLOW)
    difference = target_flow - input_flow
    epe = torch.norm(difference, p=2, dim=1)
    # valid_epe = epe[valid]
    # loss = valid_epe.mean()
    loss = epe.mean()

    '''
    metrics = {
        'epe': epe.mean().item(),
        '1px': (valid_epe < 1).float().mean().item(),
        '3px': (valid_epe < 3).float().mean().item(),
        '5px': (valid_epe < 5).float().mean().item(),
    }
    '''
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return loss, metrics

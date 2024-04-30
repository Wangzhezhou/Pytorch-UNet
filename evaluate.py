
import torch
from tqdm import tqdm
from utils.dice_score import epe_loss 

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    total_epe_loss = 0.0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, flow_true = batch['image'], batch['flow']
            
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            flow_true = flow_true.to(device=device, dtype=torch.float32)
            flow_true = flow_true.permute(0, 3, 1, 2)
            # 预测光流
            flow_pred = net(images)
        
            # 计算EPE损失
            epe = epe_loss(flow_pred, flow_true)
            total_epe_loss += epe.item()

    net.train()
    
    return total_epe_loss / max(num_val_batches, 1)

import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, KITTI
from utils.dice_score import epe_loss

dir_img = Path('./data/KITTI/image_2')
dir_flow = Path('./data/KITTI/flow_occ')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 100,
        batch_size: int = 1,
        learning_rate: float = 0.009526887711772588 * 0.1,
        val_percent: float = 0.052267383182929415,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 0.00001459482410533705,
        momentum: float = 0.8,
        gradient_clipping: float = 0.5,
):
    # 1. Create dataset
    dataset = KITTI(dir_img, dir_flow)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment = wandb.init(project='U-Net-optical-flow', resume='allow', anonymous='never')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
    '''
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=epochs * len(train_loader),  
        pct_start=0.1,
        anneal_strategy='linear'
    )
    '''
    # modify: change scheduler to Cosine
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 100, eta_min=2e-7)

    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0


    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0  
        accumulated_loss = 0
        total_1px_accuracy = 0
        total_3px_accuracy = 0.0
        total_5px_accuracy = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # images, flow, valids = batch['image'], batch['flow'], batch['valid']
                images, flow = batch['image'], batch['flow']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                flow = flow.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # valids = valids.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    flow_pred = model(images)
                    flow = flow.permute(0, 3, 1, 2)

                    print("Predicted flow shape:", flow_pred.shape)
                    print("True flow shape:", flow.shape)
                    # loss, metrics = epe_loss(flow_pred, flow, valids)
                    loss, metrics = epe_loss(flow_pred, flow)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                
                # modify 1 (add)
                # scheduler.step()
                # end modify 1
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                accumulated_loss += loss.item()
                total_1px_accuracy += metrics['1px']
                total_3px_accuracy += metrics['3px']
                total_5px_accuracy += metrics['5px']
                batch_count += 1

                if batch_count == 10:
                    average_loss = accumulated_loss / batch_count
                    average_1px_accuracy = total_1px_accuracy / batch_count
                    average_3px_accuracy = total_3px_accuracy / batch_count
                    average_5px_accuracy = total_5px_accuracy / batch_count
                    scheduler.step()
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'average train loss': average_loss,
                        '1px accuracy': average_1px_accuracy,
                        '3px accuracy': average_3px_accuracy,
                        '5px accuracy': average_5px_accuracy,
                        'step': global_step,
                        'epoch': epoch
                    })
                    epoch_loss += accumulated_loss
                    accumulated_loss = 0
                    total_1px_accuracy = 0
                    total_3px_accuracy = 0.0
                    total_5px_accuracy = 0.0
                    batch_count = 0
               
                    pbar.set_postfix(**{'loss (batch)': average_loss})

                # Evaluation round
                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if value.grad is not None:  # 只有在梯度非None时才检查是否有无穷或NaN
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_metrics = evaluate(model, val_loader, device, amp)
                        logging.info('Validation EPE score: {:.2f}'.format(val_metrics['epe']))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation epe score': val_metrics['epe'],
                                '1px accuracy': val_metrics['1px_accuracy'],
                                '3px accuracy': val_metrics['3px_accuracy'],
                                '5px accuracy': val_metrics['5px_accuracy'],
                                'images': wandb.Image(images[0].cpu()),
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except Exception as e:
                            logging.error("Error while logging to wandb: {}".format(str(e)))

               

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 120)))
            logging.info(f'Checkpoint {epoch} saved!')
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=0.009526887711772588,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=5.2267383182929415,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--momentum', type=float, default=0.8, help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.00001459482410533705, help='Weight decay for optimizer')
    parser.add_argument('--gradient_clipping', type=float, default=0.5, help='Gradient clipping threshold')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

'''
if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Create an instance of UNet
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device=device)  # Ensure the model is on the correct device

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Load the pre-trained model
    model_path = './checkpoints/checkpoint_epoch100.pth'  # Adjust to where your pre-trained model is saved
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {model_path}')

    # Start fine-tuning on KITTI dataset
    try:
        train_model(
            model=model,
            epochs=20,  # You might want to train less epochs for fine-tuning
            batch_size=args.batch_size,
            learning_rate=args.lr * 0.1,  # Reduce the learning rate for fine-tuning
            device=device,
            val_percent=args.val / 100,
            amp=args.amp,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('OutOfMemoryError during fine-tuning!')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp
        )
'''
if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Create an instance of UNet
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device=device)  # Ensure the model is on the correct device

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Load the pre-trained model
    model_path = './checkpoints/checkpoint_epoch100.pth'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {model_path}')

    # Freeze specific layers
    for name, param in model.named_parameters():
        # Freeze the initial convolutions and the first two downsampling layers
        if 'inc' in name or 'down1' in name or 'down2' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Start fine-tuning on KITTI dataset
    try:
        train_model(
            model=model,
            epochs=20,  # set the finetuning epoch to 20
            batch_size=args.batch_size,
            learning_rate=args.lr * 0.1,  # Reduced learning rate for fine-tuning
            device=device,
            val_percent=args.val / 100,
            amp=args.amp,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('OutOfMemoryError during fine-tuning!')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp
        )

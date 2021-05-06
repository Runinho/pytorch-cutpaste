# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import MVTecAT
from cutpaste import CutPaste, cut_paste_collate_fn
from model import ProjectionNet
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def run_training(data_type="screw"):
    torch.multiprocessing.freeze_support()
    # TODO: use script params for hyperparameter
    # Temperature Hyperparameter currently not used
    temperature = 0.2
    epochs = 64*4
    device = "cuda"
    pretrained = True

    weight_decay = 0.00003
    learninig_rate = 0.03
    momentum = 0.9
    #TODO: use f strings also for the date LOL
    model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )

    #augmentation:
    size = 256
    min_scale = 0.5

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    #TODO: we might want to normalize the images.

    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.Resize((256,256)))
    train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    train_transform.transforms.append(CutPaste(transform = after_cutpaste_transform))
    # train_transform.transforms.append(transforms.ToTensor())

    train_data = MVTecAT("Data", data_type, transform = train_transform, size=size)
    dataloader = DataLoader(train_data, batch_size=32,
                            shuffle=True, num_workers=8, collate_fn=cut_paste_collate_fn,
                            persistent_workers=True)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs") / model_name)

    # create Model:
    model = ProjectionNet(pretrained=pretrained)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)

    step = 0
    import torch.autograd.profiler as profiler
    num_batches = len(dataloader)
    for epoch in tqdm(range(epochs)):
        for batch_idx, data in enumerate(dataloader):
            x1, x2 = data
            x1 = x1.to(device)
            x2 = x2.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            xc = torch.cat((x1, x2), axis=0)
            embeds, logits = model(xc)
            
    #         embeds = F.normalize(embeds, p=2, dim=1)
    #         embeds1, embeds2 = torch.split(embeds,x1.size(0),dim=0)
    #         ip = torch.matmul(embeds1, embeds2.T)
    #         ip = ip / temperature

    #         y = torch.arange(0,x1.size(0), device=device)
    #         loss = loss_fn(ip, torch.arange(0,x1.size(0), device=device))

            y = torch.tensor([0, 1], device=device)
            y = y.repeat_interleave(x1.size(0))
            loss = loss_fn(logits, y)
            

            # regulize weights:
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / num_batches)
            
            writer.add_scalar('loss', loss.item(), step)
            
    #         predicted = torch.argmax(ip,axis=0)
            predicted = torch.argmax(logits,axis=1)
    #         print(logits)
    #         print(predicted)
    #         print(y)
            accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
            writer.add_scalar('acc', accuracy, step)
            
            step += 1
        writer.add_scalar('epoch', epoch, step)
    torch.save(model.state_dict(), Path("models") / f"{model_name}.tch")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    args = parser.parse_args()
    print(args)
    all_types = ['bottle',
             'cable',
             'capsule',
             'carpet',
             'grid',
             'hazelnut',
             'leather',
             'metal_nut',
             'pill',
             'screw',
             'tile',
             'toothbrush',
             'transistor',
             'wood',
             'zipper']
    
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")
    
    for data_type in types:
        print(f"training {data_type}")
        run_training(data_type)
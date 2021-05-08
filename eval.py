from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset import MVTecAT
from cutpaste import CutPaste
from model import ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np

def eval_model(modelname, defect_type, device="cpu", save_plots=False, size=256, show_training_data=True, model=None):
    # create test dataset
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size,size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    test_data = MVTecAT("Data", defect_type, size, transform = test_transform, mode="test")

    dataloader_test = DataLoader(test_data, batch_size=64,
                            shuffle=False, num_workers=0)

    # create model
    if model is None:
        print(f"loading model {modelname}")
        model = ProjectionNet(pretrained=False)
        model.load_state_dict(torch.load(modelname))
        model.to(device)
        model.eval()

    #get embeddings for test data
    labels = []
    embeds = []
    with torch.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x.to(device))

            # save 
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = torch.cat(labels)
    embeds = torch.cat(embeds)

    # train data / train kde
    # TODO: put the GDE stuff into the Model class and do this at the end of the training
    test_data = MVTecAT("Data", defect_type, size, transform=test_transform, mode="train")

    dataloader_train = DataLoader(test_data, batch_size=64,
                            shuffle=False, num_workers=0)
    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            embed, logit = model(x.to(device))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)

    #create eval plot dir
    if save_plots:
        eval_dir = Path("eval") / modelname
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # plot tsne
        # also show some of the training data
        show_training_data = False
        if show_training_data:
            #augmentation settig
            # TODO: but all of this in the same place and put it into the args
            min_scale = 0.5

            # create Training Dataset and Dataloader
            after_cutpaste_transform = transforms.Compose([])
            after_cutpaste_transform.transforms.append(transforms.ToTensor())
            after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]))
            #TODO: we might want to normalize the images.

            train_transform = transforms.Compose([])
            #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
            #train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
            train_transform.transforms.append(CutPaste(transform=after_cutpaste_transform))
            # train_transform.transforms.append(transforms.ToTensor())

            train_data = MVTecAT("Data", defect_type, transform=train_transform, size=size)
            dataloader_train = DataLoader(train_data, batch_size=32,
                        shuffle=True, num_workers=8, collate_fn=cut_paste_collate_fn,
                        persistent_workers=True)
            # inference training data
            train_labels = []
            train_embeds = []
            with torch.no_grad():
                for x1, x2 in dataloader_train:
                    x = torch.cat([x1,x2], axis=0)
                    embed, logit = model(x.to(device))

                    # generate labels:
                    y = torch.tensor([0, 1])
                    y = y.repeat_interleave(x1.size(0))

                    # save 
                    train_embeds.append(embed.cpu())
                    train_labels.append(y)
                    # only less data
                    break
            train_labels = torch.cat(train_labels)
            train_embeds = torch.cat(train_embeds)

            # for tsne we encode training data as 2, and augmentet data as 3
            tsne_labels = torch.cat([labels, train_labels + 2])
            tsne_embeds = torch.cat([embeds, train_embeds])
        else:
            tsne_labels = labels
            tsne_embeds = embeds
        plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")
    else:
        eval_dir = Path("unused")
    # # estemate KDE parameters
    # # use grid search cross-validation to optimize the bandwidth
    # params = {'bandwidth': np.logspace(-10, 10, 50)}
    # grid = GridSearchCV(KernelDensity(), params)
    # grid.fit(embeds)

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    # # use the best estimator to compute the kernel density estimate
    # kde = grid.best_estimator_
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(train_embed)
    scores = kde.score_samples(embeds)
    # print(scores)
    # we get the probability to be in the correct distribution
    # but our labels are inverted (1 for out of distribution)
    # so we have to relabel 

    roc_auc = plot_roc(labels, scores, eval_dir / "roc_plot.png", modelname=modelname, save_plots=save_plots)
    

    return roc_auc
    

def plot_roc(labels, scores, filename, modelname="", save_plots=False):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    #plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc

def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=300)
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:,0], tsne_results[:,1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--model_dir', default="models",
                    help=' directory contating models to evaluate (default: models)')
    
    parser.add_argument('--cuda', default=False,
                    help='use cuda for model predictions (default: False)')

    

    args = parser.parse_args()

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
    
    device = "cuda" if args.cuda else "cpu"

    # find models
    model_names = [list(Path(args.model_dir).glob(f"model-{data_type}*"))[0] for data_type in types]
    for model_name, data_type in zip(model_names, types):
        print(f"evaluating {data_type}")

        roc_auc = eval_model(model_name, data_type, save_plots=True, device=device)
        print(f"{data_type} AUC: {roc_auc}")
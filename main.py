import data_loader
import dataset.mpii
import config
import torch
import model.deepercut

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    mpii_dataset = dataset.mpii.MPIIDataset(config)
    dataloader = data_loader.create_dataloader()
    # sample_batched = next(iter(dataloader))

    model = model.deepercut.DeeperCut(config.num_joints)
    if torch.cuda.is_available():
        model.to('cuda')

    for i_batch, sample_batched in enumerate(dataloader):
        im = sample_batched['image']
        with torch.no_grad():
            output = model(im)
        print(sample_batched['scmap'].size() == output.size())
        if i_batch == 100:
            break;

""" 
for i_batch, sample_batched in enumerate(dataloader):
    im = sample_batched['image']
    print(i_batch, im.size(),
          sample_batched['scmap'].size())
    visualizer.imshow(im)
    if i_batch == 3:
        break;
"""

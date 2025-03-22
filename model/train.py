import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from baseline_segnet import SegNet
from efficient_unet import EfficientUNet
from baseline_unet import UNet
from enum import IntEnum
from PIL import Image

img2t = transforms.ToTensor()

def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(tensor_trimap)
])

class OxfordPetsSegmentationDataset(datasets.OxfordIIITPet):
    def __init__(self, mask_transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_transform = mask_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        # Get image and target from the dataset
        (image, target) = super().__getitem__(idx)

        # Apply the mask transformation (resize and convert to tensor)
        if self.mask_transform:
            target = self.mask_transform(target)

        return image, target


class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2


def trimap2f(trimap):
    return (img2t(trimap) * 255.0 - 1) / 2



def train_model(model, trainloader, loss_class, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for id, (X_train, y_train) in enumerate(trainloader, 0):
            print(f'Training sample {id}')
            optimizer.zero_grad()

            y_hat = model(X_train)
            y_train = y_train.squeeze(dim=1)
            loss = loss_class(y_hat, y_train)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()*X_train.size(0)

            #if epoch % 10 == 9:
        print(f"Epoch {epoch+1} of {epochs}, Loss: {running_loss/len(trainloader.dataset)}")


def main():
    # check cuda availability
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)


    # prepare data
    trainset = OxfordPetsSegmentationDataset(root='./data', split='trainval', target_types='segmentation', download=True, transform=transform, mask_transform=mask_transform)
    #testset = datasets.OxfordIIITPet(root='./data', split='test', target_types= 'segmentation', download=True, transform=transform)

    # Create DataLoaders for batch processing
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True) # Using small batch-size as running out of memory
    #val_loader = DataLoader(testset, batch_size=16, shuffle=False)

    # check input X shape
    X_train_batch, _ = next(iter(trainloader))
    print(X_train_batch.shape)

    # initialise model
    #model = SegNet()
    #model = EfficientUNet()
    model = UNet(3, 3)

    # test model giving correct shape
    model.eval()
    output = model(X_train_batch)
    print(output.shape)

    # initialise optimiser & loss class
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    train_model(model, trainloader, loss_fn, optimizer, 2)


if __name__ == "__main__":
    main()
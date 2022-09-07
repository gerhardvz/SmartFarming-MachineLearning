import io

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
import torch.nn.functional as F
import pandas as pd


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}  # Combine accuracies

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# model executes if the above is exicuted.

# This works by passing in a byte[]
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        #                                     transforms.CenterCrop(224),
        transforms.ToTensor(),
        #                                     transforms.Normalize(
        #                                         [0.485, 0.456, 0.406],
        #                                         [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    # return (image).unsqueeze(0)
    return my_transforms(image).unsqueeze(0)


# for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_prediction(image_bytes):
    image = transform_image(image_bytes=image_bytes)
    image = image.to(device)
    output = model.forward(image)
    #
    probs = torch.nn.functional.softmax(output, dim=1)
    # probs = new_model(image)
    conf, classes = torch.max(probs, dim=1)
    return conf.item(), index_to_clasification[classes.item()]


def init():
    label_df = pd.read_csv('classes.csv')
    global index_to_clasification
    index_to_clasification = label_df.to_dict()['Classifications']
    # Specify a path
    PATH = "plant-disease-model-complete.pth"
    # Load
    global model
    model = torch.load(PATH)
    model.eval()
    global device
    device = get_default_device()


def test(image_path):


    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        # testImageDir.unsqueeze(0)
        conf, y_pre = get_prediction(image_bytes)
        print(y_pre, ' at confidence score:{0:.2f}'.format(conf))



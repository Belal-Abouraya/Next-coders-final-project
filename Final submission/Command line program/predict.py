#package imports

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json


def get_input_args():
    """parse the command line arguments and returns them
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", required = True, type = str,
                    help = "path to the model checkpoint")
    parser.add_argument("--image_dir", required = True, type = str,
                        help = "path to the image to be identified")
    parser.add_argument("--topk", type = int, default = 1,
                        help = "return the top k probabilities")
    parser.add_argument("--gpu", type = str, default = "True", choices = ["True", "False"],
                       help = "use the gpu for inference")
    parser.add_argument("--category_name_dir", type = str, default = None,
                        help = "path to a json file with category names")
    return parser.parse_args()

def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an tensor
    """
    #refrence: https://gist.github.com/tomvon/ae288482869b495201a0
    #resizing the image while keeping the aspect ratio 

    basewidth = 256
    wpercent = basewidth/float(image.size[0])
    height = int(image.size[1]*wpercent)
    image = image.resize((basewidth, height))
    
    #refrence: https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    #center cropping the image

    left = (image.size[0] - 224)/2
    upper = (image.size[1] - 224)/2
    right = (image.size[0] + 224)/2
    lower = (image.size[1] + 224)/2
    image = image.crop((left, upper, right, lower))
    
    #refrence: https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array
    #refrence: https://stackoverflow.com/questions/11064786/get-pixels-rgb-using-pil
    #normalizing the image

    image = image.convert("RGB")
    np_image = np.array(image)
    np_image = np_image / 255
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - means)/std
    np_image = np_image.transpose((2,0,1))
    tensor_image = torch.tensor(np_image).float()
    return tensor_image

def load_checkpoint(filepath):
    """loads a model from a checkpoint and returns it
    """
    #loading the checkpoint and creating the classifier layer

    checkpoint = torch.load(filepath)

    classifier = nn.Sequential(nn.Linear(checkpoint["input"], checkpoint["h1"]),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(checkpoint["h1"], checkpoint["output"]),
                               nn.LogSoftmax(dim=1))
    classifier.load_state_dict(checkpoint["state dict"])

    #attaching the classifier to the pretrained model

    if checkpoint["base model"] == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier = classifier
    else:
        model = models.resnet18(pretrained=True)
        model.fc = classifier
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint["class to idx"]
    return model

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    processed_image = process_image(image).unsqueeze(0)
    model.eval()
    model = model.to(device)
    model = model.float()
    processed_image = processed_image.to(device)
    probabilities = torch.exp(model.forward(processed_image))
    top_ps, top_classes = probabilities.topk(topk, dim=1)
    return top_ps, top_classes


def main():
    """the main program function
    """
    #retrieving command line argument values

    in_arg = get_input_args()
    checkpoint_dir = in_arg.checkpoint_dir
    image_dir = in_arg.image_dir
    top_k = in_arg.topk
    category_name_dir = in_arg.category_name_dir
    gpu = True if in_arg.gpu == "True" else False

    #determining the device to infere on

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda"):
        print("infering on the gpu")
    else:
        print("infering on the cpu")
    
    #loading the model and making the prediction

    model = load_checkpoint(checkpoint_dir)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_ps, top_classes = predict(image_dir, model, top_k, device)
    top_classes = top_classes.cpu().detach().numpy().astype(str)
    top_ps = top_ps.cpu().detach().numpy()

    #printing the class or category name based on the user's choice

    if category_name_dir:

        with open(category_name_dir, 'r') as f:
            cat_to_name = json.load(f)
        
        idx_to_name = {k: cat_to_name[v] for k, v in idx_to_class.items()}

        for i, v in enumerate(top_classes[0]):
            top_classes[0][i] = idx_to_name[int(v)]
    
    else:
        for i, v in enumerate(top_classes[0]):
            top_classes[0][i] = idx_to_class[int(v)]
    
    print(top_ps, top_classes)


if __name__ == "__main__":
    main()

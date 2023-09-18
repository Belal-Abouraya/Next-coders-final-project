import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np



torch.set_default_tensor_type("torch.FloatTensor")

def get_input_args():
    """parse the command line arguments and returns them
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type = str, required = True,
                       help = "path to the data folder")
    parser.add_argument("--arch", type = str, default = "vgg16", choices = ["vgg16", "resnet18"],
                      help = "select the neural network architecture")
    parser.add_argument("--save_dir", type = str, default = None, 
                        help = "location to save model")
    parser.add_argument("--learning_rate", type = float, default = 0.003,
                        help = "learning rate of the model")
    parser.add_argument("--epochs", type = int, default = 5,
                        help = "number of epochs")
    parser.add_argument("--hidden_units", type = int, default = 256,
                        help = "number of hidden units of the model")
    parser.add_argument("--gpu", type = str, default = "True", choices = ["True", "False"],
                        help = "use the gpu to accelerate the learning process")

     
    return parser.parse_args()

def get_loaders(data_dir):

    """creates the data loader objects and returns the
    """
    #set the directories for the data

    data_dir = data_dir
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"

    #defining the transfroms on the training data

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    #defining the transforms on the validation data

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    return train_dataset, train_loader, valid_loader

def train(arch, lr, epochs, hidden, device, train_loader, valid_loader):
    """creates a trained model and returns it
    """

    #create the model based on the user input
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(25088, hidden),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.2),
                                            nn.Linear(hidden, 102),
                                            nn.LogSoftmax(dim=1))
        model.classifier = classifier
        model.base_model = "vgg16" #used in saving the model
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential(nn.Linear(512, hidden),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(hidden, 102),
                                       nn.LogSoftmax(dim=1))
        model.fc = classifier
        model.base_model = "resnet18" #used in saving the model
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    #initialize variables
    steps = 0
    running_loss = 0
    print_every = 8
    running_loss = 0
    model.to(device)
    criterion = nn.NLLLoss()

    #training loop

    for e in range(epochs):
        
        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        batch_loss = criterion(output, labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"validation accuracy: {accuracy/len(valid_loader):.3f}")
                model.train()
                running_loss = 0
    
    return model 

def save_chechpoint(model, train_dataset, hidden, save_dir):
    """saves a model in a given directory
    """
    #saving the basic data about the model

    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {"h1": hidden,
                  "output" : 102,
                  "base model": model.base_model,
                  "class to idx": model.class_to_idx}

    #saving the classifier data depending on the base model used

    if model.base_model == "vgg16":
        checkpoint["input"] = 25088
        checkpoint["state dict"] = model.classifier.state_dict()
    else:
        checkpoint["input"] = 512
        checkpoint["state dict"] = model.fc.state_dict()
    
    torch.save(checkpoint, save_dir  + "/" + model.base_model + "_checkpoint.pth")
    print("model saved at:" + save_dir )


def main():
    """the main program fucntion
    """

    #retrieving the command line argument values

    in_arg = get_input_args()
    data_dir = in_arg.data_dir
    arch = in_arg.arch
    save_dir = in_arg.save_dir
    lr = in_arg.learning_rate
    epochs = in_arg.epochs
    n_hidden = in_arg.hidden_units
    gpu = True if in_arg.gpu == "True" else False
    
    #determining the device to train on

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda"):
        print("training on the gpu")
    else:
        print("training on the cpu")
    
    #getting the needed loader objects

    train_dataset, train_loader, valid_loader = get_loaders(data_dir)

    #training the model

    model = train(arch, lr, epochs, n_hidden, device, train_loader, valid_loader)

    #saving the model if the user asks

    if save_dir:
        save_chechpoint(model, train_dataset, n_hidden, save_dir)
    


if __name__ == "__main__":
    main()


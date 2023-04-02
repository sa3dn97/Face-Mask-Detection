# import libraries
import numpy as np  
import torch       
import torchvision 
import matplotlib.pyplot as plt  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device as GPU or CPU

data_dir = "Images/train"
normalize = torchvision.transforms.Normalize(    # Normalize data according to set mean and standard deviation
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

test_transforms = torchvision.transforms.Compose([ # Compose transforms to be applied on test images
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()])


def load_split_train_test(datadir, valid_size=0.2):   # Split the train and test dataset
    
    train_transforms = torchvision.transforms.Compose([              # Compose transforms to be applied on train images
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(), normalize]
    )
    test_transforms = torchvision.transforms.Compose([               # Apply same transform on test images
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(), normalize]
    )
    
    train_data = torchvision.datasets.ImageFolder(datadir,      # Load the training dataset
                                                  transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(datadir,       # Load the test dataset
                                                 transform=test_transforms)
                                                 
    num_train = len(train_data)    # Get number of images in train dataset
    indices = list(range(num_train))    # Create indices for splitting
    split = int(np.floor(valid_size * num_train))    # Calculate split size
    np.random.shuffle(indices)     # shuffle indices to randomly pick data
    # Splitting the data into training and testing sets. 22
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=64
    )
    testloader = torch.utils.data.DataLoader(
        test_data, sampler=test_sampler, batch_size=64
    )
    # Returning the trainloader and testloader from the function after initialization
    return trainloader, testloader


# Creating a PyTorch dataloader with the given train/test datasets
trainloader, testloader = load_split_train_test(data_dir, 0.2)
print(trainloader.dataset.classes)

# Initializing the VGG16 model with pretrained weights    
vggnet = torchvision.models.vgg16(pretrained=True)

# Setting requires_grad to False to freeze all parameters of the VGG16 model
for p in vggnet.parameters():
    p.requires_grad = False

# Replacing the last fully connected layer of VGG16 with our own 2-class classifier 
vggnet.classifier[6] = torch.nn.Linear(vggnet.classifier[6].in_features, 2)

# Moving the model to GPU for faster training
vggnet.to(device)

# Defining a CrossEntropyLoss for object classification problem
lossfun = torch.nn.CrossEntropyLoss()

# Defining Stochastic Gradient Descent optimizer with learning rate of 0.001 and momentum of 0.9     
optimizer = torch.optim.SGD(vggnet.parameters(), lr=0.001, momentum=0.9)

# Number of epochs to iterate through
numepochs = 3

# Initializing losses
trainLoss = torch.zeros(numepochs)
testLoss = torch.zeros(numepochs)
trainAcc = torch.zeros(numepochs)
testAcc = torch.zeros(numepochs)

# loop over epochs
for epochi in range(numepochs):

    # loop over training data batches
    vggnet.train()  # switch to train mode
    batchLoss = []
    batchAcc = []
    for X, y in trainloader:
        print('Saad')
        # push data to GPU
        X = X.to(device)
        y = y.to(device)

        # forward pass and loss
        yHat = vggnet(X)
        loss = lossfun(yHat, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss and accuracy from this batch
        batchLoss.append(loss.item())
        batchAcc.append(torch.mean((torch.argmax(yHat, axis=1) == y)
                                   .float()).item())
    # end of batch loop...

    # and get average losses and accuracies across the batches
    trainLoss[epochi] = np.mean(batchLoss)
    trainAcc[epochi] = 100 * np.mean(batchAcc)

    # test performance (here done in batches!)
    vggnet.eval()  # switch to test mode
    batchAcc = []
    batchLoss = []
    for X, y in testloader:
        
        # push data to GPU
        X = X.to(device)
        y = y.to(device)

        # forward pass and loss
        with torch.no_grad():
            yHat = vggnet(X)
            loss = lossfun(yHat, y)

        # loss and accuracy from this batch
        batchLoss.append(loss.item())
        batchAcc.append(torch.mean((
            torch.argmax(yHat, axis=1) == y).float()).item())
    # end of batch loop...

    # and get average losses and accuracies across the batches
    testLoss[epochi] = np.mean(batchLoss)
    testAcc[epochi] = 100 * np.mean(batchAcc)

    # print out a status update
print(
  f"Finished epoch {epochi+1}/{numepochs}."
  f"Test accuracy = {testAcc[epochi]:.2f}%"
)
torch.save(vggnet, "model.pth")

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(trainLoss, "s-", label="Train")
ax[0].plot(testLoss, "o-", label="Test")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss (MSE)")
ax[0].set_title("Model loss")
ax[0].legend()

ax[1].plot(trainAcc, "s-", label="Train")
ax[1].plot(testAcc, "o-", label="Test")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy (%)")
ax[1].set_title(
    f"Final model train/test accuracy: {trainAcc[-1]:.2f}/{testAcc[-1]:.2f}%"
)
ax[1].legend()

plt.suptitle("Pretrained VGG-16 on STL10 data", fontweight="bold", fontsize=14)
plt.show()

# inspect a few random images

X, y = next(iter(testloader))
X = X.to(device)
y = y.to(device)
vggnet.eval()
predictions = torch.argmax(vggnet(X), axis=1)


fig, axs = plt.subplots(4, 4, figsize=(10, 10))

for (i, ax) in enumerate(axs.flatten()):

    # extract that image (need to transpose it back to 96x96x3)
    pic = X.data[i].cpu().numpy().transpose((1, 2, 0))
    pic = pic - np.min(pic)  # undo normalization
    pic = pic / np.max(pic)

    # show the image
    ax.imshow(pic)

    # label and true class
    label = testloader.dataset.classes[predictions[i]]
    truec = testloader.dataset.classes[y[i]]
    title = f"Pred: {label}  -  true: {truec}"

    # set the title with color-coded accuracy
    titlecolor = "g" if truec == label else "r"
    ax.text(
        48,
        90,
        title,
        ha="center",
        va="top",
        fontweight="bold",
        color="k",
        backgroundcolor=titlecolor,
        fontsize=5,
    )
    ax.axis("off")

plt.tight_layout()
plt.show()


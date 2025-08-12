import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# VGG 11
class VGG11(nn.Module):
    def __init__(self, classes=10):
        super(VGG11, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        output = self.classifier(x)
        return output
    


# get test accuracy
def getTestAccuracy(network, testLoad):
    network.eval()
    numCorrect = 0
    total = 0

    with torch.no_grad():
        for images, labels in testLoad:
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            _, predicted = outputs.max(1)
            numCorrect += (predicted == labels).double().sum().item()
            total += labels.size(0)

    return numCorrect / total 

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    horizontalFlip = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
    ])

    verticalFlip = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
    ])

    smallNoise = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x : x + (0.01 ** 0.5) * torch.randn_like(x)),
    ])

    mediumNoise = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x : x + (0.1 ** 0.5) * torch.randn_like(x)),
    ])

    bigNoise = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x : x + (1 ** 0.5) * torch.randn_like(x)),
    ])

    train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    trainHorizontal = torchvision.datasets.MNIST(root='./data', train=True, transform=horizontalFlip, download=True)
    trainVertical = torchvision.datasets.MNIST(root='./data', train=True, transform=verticalFlip, download=True)
    trainNoise = torchvision.datasets.MNIST(root='./data', train=True, transform=mediumNoise, download=True)
    trainSNoise = torchvision.datasets.MNIST(root='./data', train=True, transform=smallNoise, download=True)
    test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    testHorizontal = torchvision.datasets.MNIST(root='./data', train=False, transform=horizontalFlip, download=True)
    testVertical = torchvision.datasets.MNIST(root='./data', train=False, transform=verticalFlip, download=True)
    testSmallNoise = torchvision.datasets.MNIST(root='./data', train=False, transform=smallNoise, download=True)
    testMediumNoise = torchvision.datasets.MNIST(root='./data', train=False, transform=mediumNoise, download=True)
    testBigNoise = torchvision.datasets.MNIST(root='./data', train=False, transform=bigNoise, download=True)

    trainLoad = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    trainHLoad = torch.utils.data.DataLoader(trainHorizontal, batch_size=64, shuffle=True)
    trainVLoad = torch.utils.data.DataLoader(trainVertical, batch_size=64, shuffle=True)
    trainNLoad = torch.utils.data.DataLoader(trainNoise, batch_size=64, shuffle=True)
    trainSNLoad = torch.utils.data.DataLoader(trainSNoise, batch_size=64, shuffle=True)
    testLoad = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    testHLoad = torch.utils.data.DataLoader(testHorizontal, batch_size=64, shuffle=False)
    testVLoad = torch.utils.data.DataLoader(testVertical, batch_size=64, shuffle=False)
    testSLoad = torch.utils.data.DataLoader(testSmallNoise, batch_size=4, shuffle=False)
    testMLoad = torch.utils.data.DataLoader(testMediumNoise, batch_size=4, shuffle=False)
    testBLoad = torch.utils.data.DataLoader(testBigNoise, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = VGG11(classes=10).to(device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    epochs = 3 # change epoch numbers here
    trainLosses = []
    testLosses = []
    trainAccuracies = []
    testAccuracies = []
    flipAccuracies = []
    noiseAccuracies = []
    #Change to [trainLoad, trainHLoad, trainVLoad, trainNLoad, trainSNLoad] for augmented training set (question 4)
    trainLoadList = [trainLoad]
    for epoch in range(epochs):
        network.train()
        trainLoss = 0
        numCorrect = 0
        total = 0
        for trainingLoad in trainLoadList:
            for images, labels in trainingLoad:
                images = images.to(device)
                labels = labels.to(device)
                outputs = network(images)
                loss = lossFunction(outputs, labels)
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()

                trainLoss += loss.item()
                _, predicted = outputs.max(1)
                numCorrect += (predicted == labels).double().sum().item()
                total += labels.size(0)

        trainLoss /= len(trainLoad)
        trainAccuracy = numCorrect / total
        trainLosses.append(trainLoss)
        trainAccuracies.append(trainAccuracy)
        
        #TEST ---------------------------------------------
        network.eval()
        testLoss = 0
        numCorrect = 0
        total = 0

        with torch.no_grad():
            for images, labels in testLoad:
                images = images.to(device)
                labels = labels.to(device)
                outputs = network(images)
                loss = lossFunction(outputs, labels)

                testLoss += loss.item()
                _, predicted = outputs.max(1)
                numCorrect += (predicted == labels).double().sum().item()
                total += labels.size(0)

        testLoss /= len(testLoad)
        testAccuracy = numCorrect / total
        testLosses.append(testLoss)
        testAccuracies.append(testAccuracy)

        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {trainLoss:.4f}, Train Acc: {trainAccuracy:.4f}, Test Loss: {testLoss:.4f}, Test Acc: {testAccuracy:.4f}")

    testHAccuracy = getTestAccuracy(network, testHLoad)
    testVAccuracy = getTestAccuracy(network, testVLoad)
    flipAccuracies.append(testHAccuracy)
    flipAccuracies.append(testVAccuracy)

    testSAccuracy = getTestAccuracy(network, testSLoad)
    testMAccuracy = getTestAccuracy(network, testMLoad)
    testBAccuracy = getTestAccuracy(network, testBLoad)
    noiseAccuracies.append(testSAccuracy)
    noiseAccuracies.append(testMAccuracy)
    noiseAccuracies.append(testBAccuracy)

    # Q1 Part A
    plt.plot(range(1, epochs + 1), testAccuracies)
    plt.title('Test accuracy for each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy')
    plt.xticks(range(1, epochs + 1))
    plt.show()

    # Q1 Part B
    plt.plot(range(1, epochs + 1), trainAccuracies)
    plt.title('Training accuracy for each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training accuracy')
    plt.xticks(range(1, epochs + 1))
    plt.show()

    # Q1 Part C
    plt.plot(range(1, epochs + 1), testLosses)
    plt.title('Test loss for each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.xticks(range(1, epochs + 1))
    plt.show()

    # Q1 Part D
    plt.plot(range(1, epochs + 1), trainLosses)
    plt.title('Training loss for each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.xticks(range(1, epochs + 1))
    plt.show()

    # Q3 Part E (Q4 as well if using augmented training set)
    flips = ['Horizontal Flip', 'Vertical Flip']
    plt.bar(flips, flipAccuracies)
    plt.title('Test accuracy for each flip')
    plt.xlabel('Flips')
    plt.ylabel('Test accuracy')
    plt.show()

    # Q3 Part F (Q4 as well if using augmented training set)
    noises = ['0.01', '0.1', '1']
    plt.bar(noises, noiseAccuracies)
    plt.title('Test accuracy for each Gaussian Noise')
    plt.xlabel('Gaussian Noises')
    plt.ylabel('Test accuracy')
    plt.show()



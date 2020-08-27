import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchsample
# from matplotlib import pyplot

class InceptionPiece(nn.Module):
    def __init__(self,input,out):
        super(InceptionPiece, self).__init__()
        bn_size= 5
        self.bottleneck = nn.Conv2d(input,bn_size,kernel_size=1)
        self.conv1 = nn.Conv2d(bn_size,out,kernel_size=1)
        self.conv2 = nn.Conv2d(bn_size,out, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(bn_size,out, kernel_size=5,padding=2)
        self.bn = nn.BatchNorm2d(out*3)

class InceptionLikeNet(nn.Module):
    def __init__(self,ch):
        super(InceptionLikeNet, self).__init__()
        self.conv1 = nn.Conv2d(ch, 15, kernel_size=1)
        self.Incept1 = InceptionPiece(15,7)
        self.Incept2 = InceptionPiece(21,7)
        self.Incept3 = InceptionPiece(21,5)
        self.fc1 = nn.Linear(735, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.Incept1(x)
        x = F.max_pool2d(x,2)
        x = self.Incept2(x)
        x = self.Incept3(x)
        x = x.view(-1, 735)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)



class BatchNormNet(nn.Module):
    def __init__(self,ch):
        super(BatchNormNet, self).__init__()
        self.conv1 = nn.Conv2d(ch, 15, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(15)
        self.conv2 = nn.Conv2d(15, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(180, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(F.max_pool2d(self.conv4(x),2))
        x = self.bn4(x)
        x = x.view(-1, 180)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


def transfer_rotated_net_ex():

    device = torch.device('cpu')  # or 'cpu' cuda

    # here we set up some data preparation using pytorch utils.
    # Note the augmentation!
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (
    0.5,))])  # transforms.RandomAffine(degrees = 25, scale=(0.95,1.05)),
    # Transformation for the test set should not contain augmentation.
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


    batch_size = 256
    trainset = torchvision.datasets.mnist.MNIST(root='./data', train=True,
                                                download=True, transform=transform)

    # splitting a small portion of the data for validation
    val_samples = 1024
    trainset, val_set = torch.utils.data.random_split(trainset, [len(trainset) - val_samples, val_samples])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    validloader = torch.utils.data.DataLoader(val_set, batch_size=1024,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.mnist.MNIST(root='./data', train=False,
                                               download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    # 90 degrees rotation train/test sets

    transform_90 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (
        0.5,)), torchsample.transforms.Rotate(90)]) # lambda img: transforms.functional.rotate(img, 90)]
    transform_test_90 = transforms.Compose([transforms.ToTensor(), # , transforms.ToPILImage()
                                            transforms.Normalize((0.5,), (0.5,)),
                                            torchsample.transforms.Rotate(90)])

    trainset_90 = torchvision.datasets.mnist.MNIST(root='./data', train=True,
                                                download=True, transform=transform_90)

    trainset_90, val_set_90 = torch.utils.data.random_split(trainset_90, [len(trainset_90) - val_samples, val_samples])

    trainloader_90 = torch.utils.data.DataLoader(trainset_90, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    validloader_90 = torch.utils.data.DataLoader(val_set_90, batch_size=1024,
                                              shuffle=True, num_workers=1)

    testset_90 = torchvision.datasets.mnist.MNIST(root='./data', train=False,
                                               download=True, transform=transform_test_90)
    testloader_90 = torch.utils.data.DataLoader(testset_90, batch_size=batch_size,
                                             shuffle=False, num_workers=0) # num_workers=1



    net = BatchNormNet(1).to(device)

    print('\nStart training “rotated” CNN on a rotated test dataset\n')

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters())
    num_mini = 20
    acc_hist = []
    for epoch in range(15):  # loop over the dataset multiple times
        correct = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader_90, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs.to(device))
            # loss
            loss = criterion(outputs, labels.to(device))
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # some status output
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.to(device).data.view_as(pred)).sum()

            running_loss += loss.item()
            if i % num_mini == num_mini - 1:  # print every 200 mini-batches
                acc = correct.to('cpu').numpy() / num_mini / batch_size
                correct = 0
                correct_val = 0
                with torch.no_grad():  # we don't need to waste time for estimation of the gradients
                    for data, target in validloader_90:
                        output = net(data.to(device))
                        pred = output.data.max(1, keepdim=True)[1]
                        correct_val += pred.eq(target.to(device).data.view_as(pred)).sum()
                    val_accuracy = (1.0 * correct_val / len(validloader_90.dataset)).to('cpu')
                acc_hist.append([acc, val_accuracy])
                print('[%d, %5d] loss: %.3f, accuracy: %.3f, val_acc: %.3f' % (
                epoch + 1, i + 1, running_loss / num_mini, acc, val_accuracy))
                running_loss = 0.0

    print('Finished Training')
    plt.plot(acc_hist)
    plt.show()

    testlosses_90 = []
    # When using the model after training we want it's batch norm and dropout layers work in infrerence mode, not training one.
    net.to('cpu').eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # we don't need to waste time for estimation of the gradients
        for data, target in testloader_90:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader_90.dataset)
    testlosses_90.append(test_loss)
    print('\nTest set rotated 90: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader_90.dataset),
        100. * correct / len(testloader_90.dataset)))

    testlosses = []
    # When using the model after training we want it's batch norm and dropout layers work in infrerence mode, not training one.
    net.to('cpu').eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # we don't need to waste time for estimation of the gradients
        for data, target in testloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    testlosses.append(test_loss)
    print('\nTest set normal images: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    # save the net
    torch.save(net.state_dict(), 'model')

    # delete and redefine the net
    del net
    net = BatchNormNet(1).to(device)

    # load the weight
    net.load_state_dict(torch.load('model'))

     # A) freeze the entire model except for the last layer and retrain only it

    trainable_parameters = []
    for name, p in net.named_parameters():
        if "fc2" in name:
            trainable_parameters.append(p)

    # train again
    print('\nStart retraining “rotated” CNN on a normal test dataset with frozen the entire model except for the last layer\n')
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.001, momentum=0.9)
    num_mini = 20
    acc_hist = []
    for epoch in range(5):  # loop over the dataset multiple times
        correct = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs.to(device))
            # loss
            loss = criterion(outputs, labels.to(device))
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # some status output
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.to(device).data.view_as(pred)).sum()

            running_loss += loss.item()
            if i % num_mini == num_mini - 1:  # print every 200 mini-batches
                acc = correct.to('cpu').numpy() / num_mini / batch_size
                correct = 0
                correct_val = 0
                with torch.no_grad():  # we don't need to waste time for estimation of the gradients
                    for data, target in validloader:
                        output = net(data.to(device))
                        pred = output.data.max(1, keepdim=True)[1]
                        correct_val += pred.eq(target.to(device).data.view_as(pred)).sum()
                    val_accuracy = (1.0 * correct_val / len(validloader.dataset)).to('cpu')
                acc_hist.append([acc, val_accuracy])
                print('[%d, %5d] loss: %.3f, accuracy: %.3f, val_acc: %.3f' % (
                epoch + 1, i + 1, running_loss / num_mini, acc, val_accuracy))
                running_loss = 0.0

    print('Finished Training the last layer')
    plt.plot(acc_hist)
    plt.show()

    testlosses = []
    # When using the model after training we want it's batch norm and dropout layers work in infrerence mode, not training one.
    net.to('cpu').eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # we don't need to waste time for estimation of the gradients
        for data, target in testloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    testlosses.append(test_loss)
    print('\nTest set Training the last layer: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    # B) freeze all convolutional layers that extract features and retrain a classifier part

    # delete and redefine the net
    del net
    net = BatchNormNet(1).to(device)

    # load the weight
    net.load_state_dict(torch.load('model'))

    trainable_parameters = []
    for name, p in net.named_parameters():
        if "conv" in str(name):
            trainable_parameters.append(p)

    # train again
    print(
        '\nStart retraining “rotated” CNN on a normal test dataset with frozen all convolutional layers that extract features\n')
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.001, momentum=0.9)
    num_mini = 20
    acc_hist = []
    for epoch in range(5):  # loop over the dataset multiple times
        correct = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs.to(device))
            # loss
            loss = criterion(outputs, labels.to(device))
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # some status output
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.to(device).data.view_as(pred)).sum()

            running_loss += loss.item()
            if i % num_mini == num_mini - 1:  # print every 200 mini-batches
                acc = correct.to('cpu').numpy() / num_mini / batch_size
                correct = 0
                correct_val = 0
                with torch.no_grad():  # we don't need to waste time for estimation of the gradients
                    for data, target in validloader:
                        output = net(data.to(device))
                        pred = output.data.max(1, keepdim=True)[1]
                        correct_val += pred.eq(target.to(device).data.view_as(pred)).sum()
                    val_accuracy = (1.0 * correct_val / len(validloader.dataset)).to('cpu')
                acc_hist.append([acc, val_accuracy])
                print('[%d, %5d] loss: %.3f, accuracy: %.3f, val_acc: %.3f' % (
                epoch + 1, i + 1, running_loss / num_mini, acc, val_accuracy))
                running_loss = 0.0

    print('Finished Training a classifier part')
    plt.plot(acc_hist)
    plt.show()

    testlosses = []
    # When using the model after training we want it's batch norm and dropout layers work in infrerence mode, not training one.
    net.to('cpu').eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # we don't need to waste time for estimation of the gradients
        for data, target in testloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    testlosses.append(test_loss)
    print('\nTest set a classifier part: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    # C) don’t freeze any layers, but start training of the new CNN with weights of the previous model

    # delete and redefine the net
    del net
    net = BatchNormNet(1).to(device)

    # load the weight
    net.load_state_dict(torch.load('model'))

    # train again
    print(
        '\nStart retraining “rotated” CNN on a normal test dataset. Training of the new CNN with weights of the previous model\n')
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters())
    num_mini = 20
    acc_hist = []
    for epoch in range(5):  # loop over the dataset multiple times
        correct = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs.to(device))
            # loss
            loss = criterion(outputs, labels.to(device))
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # some status output
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.to(device).data.view_as(pred)).sum()

            running_loss += loss.item()
            if i % num_mini == num_mini - 1:  # print every 200 mini-batches
                acc = correct.to('cpu').numpy() / num_mini / batch_size
                correct = 0
                correct_val = 0
                with torch.no_grad():  # we don't need to waste time for estimation of the gradients
                    for data, target in validloader:
                        output = net(data.to(device))
                        pred = output.data.max(1, keepdim=True)[1]
                        correct_val += pred.eq(target.to(device).data.view_as(pred)).sum()
                    val_accuracy = (1.0 * correct_val / len(validloader.dataset)).to('cpu')
                acc_hist.append([acc, val_accuracy])
                print('[%d, %5d] loss: %.3f, accuracy: %.3f, val_acc: %.3f' % (
                epoch + 1, i + 1, running_loss / num_mini, acc, val_accuracy))
                running_loss = 0.0

    print('Finished Training previous model')
    plt.plot(acc_hist)
    plt.show()

    testlosses = []
    # When using the model after training we want it's batch norm and dropout layers work in infrerence mode, not training one.
    net.to('cpu').eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # we don't need to waste time for estimation of the gradients
        for data, target in testloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    testlosses.append(test_loss)
    print('\nTest set previous model: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))



if __name__ == '__main__':
    transfer_rotated_net_ex()
import torch
import snntorch as snn
from snntorch import functional as SF


def test_accuracy(config, net, testloader, device="cpu"):


    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # .permute(1, 0, 2, 3, 4)

            outputs, _ = net(images.permute(1, 0, 2, 3, 4))
            accuracy = SF.accuracy_rate(outputs, labels)

            total += labels.size(0)
            correct += accuracy * labels.size(0)
    
    return 100 * correct / total
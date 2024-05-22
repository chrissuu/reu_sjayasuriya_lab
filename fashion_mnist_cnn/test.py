import torch

class Tester:

    def __init__(self, test_dataloader, net):

        self.test_dataloader = test_dataloader
        self.net = net

    def test(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data

                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {len(self.test_dataloader)} test images: {100 * correct // total} %')



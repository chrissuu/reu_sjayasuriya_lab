class Trainer:

    def __init__(self, epochs, train_dataloader, criterion, optimizer):
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, net):

        for epoch in range(self.epochs):

            running_loss = 0.0

            for i, data in enumerate(self.train_dataloader, 0):

                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch+1}, {i+5:d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0
        

import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        x = self.linear(x)
        return F.softmax(x, dim=1)

def train(model, dataloader, criteria, optimizer, epochs):
    model.train()
    print("Training")
    for epoch in range(epochs):
        train_loss = []
        pred = []
        
        print("===== Epoch {}/{} =====".format(epoch+1, epochs))
        with tqdm(total=len(dataloader.dataset)) as pbar:
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(data)

                # Calculate loss
                loss = criteria(output, target)
                train_loss.append(loss)

                # Calculate accuracy from softmax
                output = [torch.argmax(result).item() for result in output]
                print(output)
                break
                # pred.append(1 if output == target else 0)

                # Backward process
                loss.backward()
                optimizer.step()
                pbar.update(len(data))

        break
        train_loss = sum(train_loss) / len(train_loss)

        print('Loss: {:.6f}'.format(train_loss.item()))
        # print('Accuracy: {:.2f}'.format(sum(pred) / len(dataloader.dataset)))

def eval(model, dataloader, criteria):
    model.eval()
    test_loss = []
    predict = []
    print("Evaluation")
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)

            # Calculate loss
            loss = criteria(output, target)
            test_loss.append(loss)

            # Calculate accuracy from softmax output
            output = torch.argmax(output).item()
            predict.append(1 if output == target else 0)
        
        test_loss = sum(test_loss) / len(dataloader.dataset)
        print('Loss: {:.6f}'.format(test_loss.item()))
        print('Accuracy: {:.2f}'.format(sum(predict) / len(dataloader.dataset)))
            
        
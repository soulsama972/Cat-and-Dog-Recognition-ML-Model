import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from math import floor
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False
MODEL_NAME = f"model-{int(time.time())}"
WIDTH = 80
HEIGHT = 80
BATCH_SIZE = 100
EPOCHS = 10
TEST_PCT = 0.1
DOG_INDEX = 0
CAT_INDEX = 1




class DogVsCats():
    cats = r'PetImages\Cat'
    dogs = r'PetImages\Dog'
    labels = {cats: 0, dogs: 1}
    traning_data = []
    catCount = 0
    dogCount = 0

    def make_traning_data(self, new_file_name_to_be_saved: str):
        for label in self.labels:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (WIDTH, HEIGHT))
                    self.traning_data.append([np.array(img), np.eye(2)[self.labels[label]]])
                    
                    if label == self.cats:
                        self.catCount += 1
                    elif label == self.dogs:
                        self.dogCount += 1
                except Exception:
                    pass
        
        np.random.shuffle(self.traning_data)
        np.save(new_file_name_to_be_saved, self.traning_data)
        
        print("Cats ", self.catCount)
        print("Dogs ", self.dogCount)



class Net(nn.Module):
    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        super().__init__()
        shape_x, shape_y = self._cal_shape(width, height, [[5, 5], [5, 5], [5, 5]])
        self.to_linear = 128 * shape_x * shape_y
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 2)
 
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("running on gpu")

        else:
            self.device = torch.device("cpu")
            print("running on cpu")
        
        self.to(self.device)

    def _cal_shape(self, w: int, h: int, kernel_size_arr: List[List[int]]):
        out_w = w
        out_h = h
        for i in kernel_size_arr:
            out_w = floor((out_w - (i[0] - 1)) / 2)
            out_h = floor((out_h - (i[1] - 1)) / 2)
        return out_w, out_h

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def forward(self, x: Union[np.ndarray, torch.Tensor]):
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32, device=self.device).view(-1, 1, WIDTH, HEIGHT)
        else:
            x = x.to(self.device).view(-1, 1, WIDTH, HEIGHT)
        X = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2, 2))
        X = F.relu(self.fc1(X.view(-1, self.to_linear)))
        X = self.fc2(X)

        return F.softmax(X, dim=1)

    def fwd_pass(self, x, y, train=False):
        if train:
            self.zero_grad()
        outputs = self(x.to(self.device))
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
        acc = matches.count(True) / len(matches)
        loss = self.loss_function(outputs, y.to(self.device))

        if train:
            loss.backward()
            self.optimizer.step()
        return acc, loss


def test(net: Net, test_x: np.ndarray, test_y: np.ndarray, batch_size=32):
    randomStart = np.random.randint(len(test_x) - batch_size)
    x, y = test_x[randomStart: randomStart + batch_size], test_y[randomStart: randomStart + batch_size]
    with torch.no_grad():
        val_acc, val_loss = net.fwd_pass(x.view(-1, 1, WIDTH, HEIGHT), y, False)
    return val_acc, val_loss


def train(net: Net, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
    with open("model.log", "a") as f:
        for _ in range(EPOCHS):
            for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
                batch_X = train_x[i:i + BATCH_SIZE].view(-1, 1, WIDTH, HEIGHT)
                batch_y = train_y[i:i + BATCH_SIZE]

                acc, loss = net.fwd_pass(batch_X, batch_y, train=True)

                if i % 50 == 0:
                    val_acc, val_loss = test(net, test_x, test_y)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")


def main():
    if REBUILD_DATA:
        dogsvscats = DogVsCats()
        dogsvscats.make_traning_data()

    net = Net()

    training_data = np.load('traningData.npy', allow_pickle=True)

    X = torch.tensor([i[0] for i in training_data]).view(-1, WIDTH, HEIGHT)
    X = X / 255.0
    Y = torch.tensor([i[1] for i in training_data], dtype=torch.float32)

    test_size = int(len(X) * TEST_PCT)

    train_x = X[:-test_size]
    train_y = Y[:-test_size]

    test_x = X[-test_size:]
    test_y = Y[-test_size:]

    train(net, train_x, train_y, test_x, test_y)
    net.save("model")




if __name__ == "__main__":
    main()

! pip install -v theano==0.9.0
! pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
#! pip install -v pandas==0.18.1

! pip install torch
! pip install torchvision

!source cuda11.1
# To see Cuda version in use
!nvcc -V
!pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root = "./data",
                       train = True,
                       download = True,
                       transform = tensor_transform)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                    batch_size = 32,
                                    shuffle = True)
                                    
class autoencoder(torch.nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28,128), 
            torch.nn.ReLU(True), #what does true do?
            torch.nn.Linear(128,64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64,36),
            torch.nn.ReLU(True),
            torch.nn.Linear(36,18),
            torch.nn.ReLU(True),
            torch.nn.Linear(18,9)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9,18),
            torch.nn.ReLU(True),
            torch.nn.Linear(18,36),
            torch.nn.ReLU(True),
            torch.nn.Linear(36,64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64,128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128,28*28),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
model = autoencoder()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3, 
                             weight_decay = 1e-5)
epochs = 3
outputs = []
losses = []
for epoch in range(epochs):
    for (image, _) in loader:
        image = image.reshape(-1,28*28)
        reconstructed = model(image)
        loss = criterion(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    outputs.append((epochs, image, reconstructed))

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses[-100:])

item = image[31].reshape(-1,28,28)
plt.imshow(item[0])

item = reconstructed[31].reshape(-1,28,28)
item = item.detach().numpy()
plt.imshow(item[0])

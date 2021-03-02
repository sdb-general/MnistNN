import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import gzip, pickle
from mnistunpacker import load_train_wrapper, load_test_wrapper
import random
import matplotlib.pyplot as plt

images, labels = load_train_wrapper()
labels = labels.squeeze(1)

class net(nn.Module):
	def __init__(self):
		super(net,self).__init__()
		self.conv1 = nn.Conv2d(3,6,5) #sizes of iflters
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 4 * 4, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		#print(x.shape, 'line 29')
		x = self.pool(F.relu(self.conv2(x)))
		#print(x.shape)
		x = x.view(-1, 16 * 4 * 4) #-1 is the batch size and setting it to -1 is good for now
		#print(x.shape)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x.squeeze(1)	
	def validate(self, x, y):
		a = self.forward(x)
		a = torch.argmax(a, dim = 1)
		b = torch.max(y, 1)[1]
		return int(torch.sum(a == b))


#start the network 

mynet = net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mynet.parameters(), lr=0.001, momentum=0.9)

#image = torch.randn(1, 3,28,28)
#print(image[0].shape)

#x = torch.Tensor(training_data[0][0])
#x.reshape()
#print(x.shape)
batchsdf = images[0:50]
#print('sdf', mynet(batchsdf).shape, labels.shape)

somelabels = labels[0:50]
#print('sds', torch.max(somelabels, 1)[1].shape)
batch_size = 50

def train(epochs):
	batches = []
	for i in range(len(images) // batch_size):
		batches.append((images[batch_size*i:batch_size*(i+1)],
						labels[batch_size*i:batch_size*(i+1)]))
	print(len(batches))
	for epoch in range(epochs):
		running_loss = 0.0
		#we need to first make the batches
		#training_data is a list of labelled images
		
		#random.shuffle(batches)

		for i, data in enumerate(batches, 0):
			inputs, exp_labels = data
			#exp_labels = exp_labels.squeeze(1)
			#print(exp_labels.shape)
			#exp_labels.squeeze_()
			#print(exp_labels.shape)
			optimizer.zero_grad()

			#forward, backward,optimise

			outputs = mynet(inputs)
			#print(exp_labels.squeeze(1).shape)
			loss = criterion(outputs, torch.max(exp_labels, 1)[1])
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if i % 200 == 199:
				print('epoch number {0},{1}, loss is {2} '
					.format(epoch +1, i, running_loss / 200))

				running_loss = 0.0



mypath = './torchtest.pth'

'''train(3)


torch.save(mynet.state_dict(), mypath)'''

mynet.load_state_dict(torch.load(mypath))

x = mynet(batchsdf)
a = torch.argmax(x, dim = 1)
b = torch.max(somelabels, 1)[1]
#print(a == b)

#print(int(torch.sum(a == b)))

#print(mynet.validate(batchsdf, somelabels))
#print(x == somelabels)

t_i, t_l = load_test_wrapper()
t_l = t_l.squeeze(1)
#print(mynet.validate(t_i, t_l))

x = t_i[10]
mynet.eval()
x.requires_grad_()
scores = mynet(x.unsqueeze(0))

score_max_index = scores.argmax()
score_max = scores[0,score_max_index]

score_max.backward()
saliency, _ = torch.max(x.grad.data.abs(),dim=0)

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))

ax1.imshow(saliency, cmap=plt.cm.hot)
#print(type(fig))
t = torch.transpose(t_i[10], 0,2)
t = torch.transpose(t, 0, 1)
ax2.imshow(t)
plt.show()
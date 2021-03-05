import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
import torchvision.datasets as dset


class net(nn.Module):
	def __init__(self):
		super(net,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
		self.linear1 = nn.Linear(7 * 7 * 64, 128)
		self.linear2 = nn.Linear(128, 10)
		self.dropout = nn.Dropout(p = 0.5)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.max_pool2d(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.max_pool2d(x)
		x = x.reshape(x.size(0), -1)
		x = self.linear1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.linear2(x)
		return x
	def validate(self, testdata): #testdata is a dataloader object
		total = 0
		correctPreds = 0
		for data in testdata:
			x, y = data
			x = x.unsqueeze(1).float()
			z = self.forward(x)
			z = torch.argmax(z, 1)
			total += len(y)
			correctPreds += int(sum(z == y))
		return correctPreds, total


'''
get the data in 
'''
batch_size = 50
mnist_trainset = dset.MNIST(root='.././data', train=True, download=True, transform=None)
train_data = mnist_trainset.data
dataset = torch.utils.data.TensorDataset(train_data, mnist_trainset.train_labels) #divide training data by 256 if necessary
dataloader = torch.utils.data.DataLoader(dataset,
										batch_size=batch_size,
										shuffle=True)

mnist_testset = dset.MNIST(root='.././data', train=False, download=True, transform=None)
dataset = torch.utils.data.TensorDataset(mnist_testset.data, mnist_testset.test_labels) #divide training data by 256 if necessary
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

#start the network

mynet = net()

criterion = nn.MSELoss()
optimizer = optim.Adam(mynet.parameters(), lr=0.004)

randomvariable = 1 #please end here


def train(epochs):

	#print(len(batches))
	for epoch in range(epochs):
		running_loss = 0.0
		#we need to first make the batches
		#training_data is a list of labelled images
		
		#random.shuffle(batches)

		for i, data in enumerate(dataloader, 0):
			image, labels = data[0].unsqueeze(1).type(torch.FloatTensor), data[1]
			optimizer.zero_grad()

			#forward, backward,optimise

			outputs = mynet(image)
			labels = torch.nn.functional.one_hot(labels, num_classes=10).type(torch.FloatTensor)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			#running_loss += loss.item()

		#end of epoch, validate
		with torch.no_grad():
			guesses, total = mynet.validate(test_dataloader)
			print('at the end of epoch {0}, we got {1} out of {2} correct'.format(epoch +1,
																			   guesses, total))



mypath = './torchtest.pth'

train(30)


torch.save(mynet.state_dict(), mypath)

#mynet.load_state_dict(torch.load(mypath))

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
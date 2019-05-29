import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import numpy as np
from UCF101 import *
from UCF101_test import *
from clstm_network_32 import *
import time

torch.backends.cudnn.benchmark = True

root_list = r'videoclips'
train_info_list = r'TrainTestlist/trainlist.txt'
test_info_list = r'TrainTestlist/testlist.txt'

#Transfrom = transforms.Compose([ClipSubstractMean(), Rescale(), ToTenor()])
Transfrom = transforms.Compose([Rescale(), ToTenor(), Normalize()])

# Training set
myUCF101_train = UCF101(train_info_list, root_list, transform = Transfrom)
trainloader = DataLoader(myUCF101_train, batch_size = 16, shuffle = True, num_workers = 0)
myUCF101_test = UCF101_test(test_info_list, root_list, transform = Transfrom)
testloader = DataLoader(myUCF101_test, batch_size = 1, shuffle = False, num_workers = 0)

#for i_batch, sample_batch in enumerate(trainloader):
#    print (i_batch, sample_batch['video_x'].size(), sample_batch['video_label'].size())
#    print (i_batch, sample_batch['video_x'], sample_batch['video_label'])

#print ('\n')

clstmNet = CNet().cuda()
clstmNet.weights_init()

init_learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clstmNet.parameters(), lr = init_learning_rate, weight_decay = 0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5,250], gamma = 0.1)

losses = []
acc_tr = []
acc_te = []
x_axis_1 = []
x_axis_2 = []
x_count_1 = 0
x_count_2 = 0

epochs = 250
for epoch in range(epochs): # Loop over the dataset multiple times
    
    clstmNet.train()
    running_loss = 0.0
    correct_tr = 0
    total_tr = 0
    correct_te = 0
    total_te = 0

    print(optimizer.state_dict)

    for i_batch, sample_batch in enumerate(trainloader):
        
        if(i_batch % 9 == 0):
            start_tr = time.time()
        
        # Get the inputs
        input_tr = sample_batch['video_x'].float()
        label_tr = sample_batch['video_label'].long()
        label_tr = label_tr.squeeze()
        label_tr -= 1

        # Zero the parameter gradients
        optimizer.zero_grad

        # Forward + Backward + Optimize
        output_tr = clstmNet.forward(input_tr)
        prediction_tr = clstmNet.predict(output_tr)
        loss = criterion(output_tr, label_tr)
        loss.backward()
        optimizer.step()
        
        #losses.append(loss.item())

        # Print statistics
        running_loss += loss.item()
        total_tr += label_tr.size(0)
        correct_tr += (prediction_tr == label_tr).double().sum().item()
        if(i_batch % 9 == 8):
            end_tr = time.time()
            running_loss /= 9
            x_count_1 += 9
            losses.append(running_loss)
            x_axis_1.append(x_count_1)
            print('[%d, %5d] loss: %.4f time: %.4fs' % (epoch+1, i_batch+1, running_loss, (end_tr - start_tr)))
            running_loss = 0.0 
    
    scheduler.step()

    clstmNet.eval()
    for index, samples in enumerate(testloader):

        probability_te = torch.zeros(len(samples), num_cls).cuda()
        for i, sample in enumerate(samples):

            input_te = sample['video_x'].float()
            label_te = sample['video_label'].long()
            label_te = label_te.squeeze(1)
            label_te -= 1

            with torch.no_grad():
                output_te = clstmNet.forward(input_te)
                prob_te = F.softmax(output_te, dim = 1)
        
            print('[sample %d, clip %d]: ' % (index+1, i+1))
            print('  label:', label_te)    
            print('  prob :', prob_te)
            print('')

            probability_te[i,:] = prob_te
    
        probability_te = torch.sum(probability_te, 0) / len(samples)
        _, prediction_te = torch.max(torch.unsqueeze(probability_te, 0), 1)
        print('Final pred: ', prediction_te)
        print('')

        total_te += label_te.size(0)
        correct_te += (prediction_te == label_te).double().sum().item()

    accuracy_tr = 100 * correct_tr / total_tr
    accuracy_te = 100 * correct_te / total_te
    acc_tr.append(accuracy_tr)
    acc_te.append(accuracy_te)
    x_count_2 += 1
    x_axis_2.append(x_count_2)
    print('Accuracy of the network on train set: %.3f %%' % (accuracy_tr))
    print('Accuracy of the network on test set: %.3f %%' % (accuracy_te))

torch.save(clstmNet.state_dict(), 'weights/clstm_weights_4_11.pkl')
print('Finished Training')

plt.subplot(1,2,1)
plt.plot(x_axis_1, losses, color = 'red')
plt.xlabel('iteration times')
plt.ylabel('loss')

plt.subplot(1,2,2)
plt.plot(x_axis_2, acc_tr, label = 'acc_tr', color = 'blue')
plt.plot(x_axis_2, acc_te, label = 'acc_te', color = 'green')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')

file = open('data_11.txt','w')
file.write(str(acc_tr))
file.write(str(acc_te))
file.close()

plt.show()
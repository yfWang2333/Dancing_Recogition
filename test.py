import torch
import torchvision
from torchvision import transforms
import numpy as np
from UCF101_test import *
from clstm_network_32 import *
import time
from collections import Counter

root_list = r'videoclips'
test_info_list = r'TrainTestlist/testlist.txt'

#Transfrom = transforms.Compose([ClipSubstractMean(), Rescale(), ToTenor()])
Transfrom = transforms.Compose([Rescale(), ToTenor(), Normalize()])

# Test set
myUCF101_test = UCF101_test(test_info_list, root_list, transform = Transfrom)
testloader = DataLoader(myUCF101_test, batch_size = 1, shuffle = False, num_workers = 0)

clstmNet = CNet().cuda()
clstmNet.load_state_dict(torch.load('weights/clstm_weights_4_11.pkl'))
clstmNet.eval()

#file = open('input1.txt','w')
correct = 0
total = 0
for index, samples in enumerate(testloader):

    #predictions = []
    probability = torch.zeros(len(samples), num_cls).cuda()
    for i, sample in enumerate(samples):

        inputs = sample['video_x'].float()
        label = sample['video_label'].long()
        label = label.squeeze(1)
        label -= 1

        #file.write(str(inputs))

        with torch.no_grad():
            output = clstmNet.forward(inputs)
            prob = F.softmax(output, dim = 1)
        
        print('[sample %d, clip %d]: ' % (index+1, i+1))
        print('  label:', label)    
        print('  prob :', prob)
        print('')

        #prediction = pred.cpu().numpy()
        #prediction = str(prediction[0]) 
        #predictions.append(prediction)

        probability[i,:] = prob
        
    #num_counting = Counter(predictions)
    #top_1 = num_counting.most_common(1)
    #print(type(top_1))
    
    probability = torch.sum(probability, 0) / len(samples)
    _, prediction = torch.max(torch.unsqueeze(probability, 0), 1)
    print('Final pred: ', prediction)
    print('')

    total += label.size(0)
    correct += (prediction == label).double().sum().item()

print('Accuracy of the network on test set: %.3f %%' % (100 * correct / total))
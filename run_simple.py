import torch
from data_processing import *
from torchvision.datasets import MNIST
from torchvision.transforms import *
from torch.utils.data import DataLoader
from nets import *
from torch import optim, nn
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn.functional as f

batch_size = 8
img_size = 32
num_augments = 1

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

test_dataset = MNIST("./data/", train=True,
                     transform=Compose([RandomResizedCrop(img_size), ToTensor(), Normalize((0.1307,), (0.3081,))]),
                     download=True)


class duplicator_dataset(Dataset):
    def __init__(self, dataset, num_repeats):
        super(duplicator_dataset, self).__init__()
        self.wrapped_dataset = dataset
        self.num_repeats = num_repeats

    def __getitem__(self, item):
        return torch.cat(tuple(self.wrapped_dataset[item][0].unsqueeze(0) for i in range(self.num_repeats)))

    def __len__(self):
        return self.wrapped_dataset.__len__()


duplicated_dataset = duplicator_dataset(test_dataset, num_augments)

loader = DataLoader(duplicated_dataset, batch_size=batch_size, shuffle=True)

labels_one_hot = torch.eye(batch_size, dtype=torch.long)
labels_one_hot = torch.repeat_interleave(labels_one_hot, num_augments, dim=0).to(device)

labels_index = torch.arange(batch_size,device=device)
labels_index = torch.repeat_interleave(labels_index, num_augments, dim=0)

#Utility function to train with labels by data augmentation
def rebatch_with_labels(input):
    global labels
    input = input.view(batch_size * num_augments, 1, img_size, img_size)
    return input

#log data
writer = SummaryWriter(f"logs/run_{time.time()}")
run_avg = RunningAvg()

# Instantiate the networks
encoder = Encoder(0.9, device)
visual_head_first = VisualHeadFirst(device, batch_size)
visual_head_second = VisualHeadSecond(device, batch_size)
fine_tuning_head = Fine_Tuning_Head(device)

feature_learner_first = Feature_Learning(encoder, visual_head_first, device)
feature_learner_second = visual_head_second #just an alias
final_classifier = Final_Classifier(encoder, fine_tuning_head,device)

#Optimise the visual head only
visual_head_optimizer = optim.SGD(list(visual_head_first.parameters()) + list(visual_head_second.parameters()), 0.01,0.9, 0.1)
visual_head_optimizer = optim.Adam(list(visual_head_first.parameters()) + list(visual_head_second.parameters()), 0.01)

#Optimise the encoder with the visual head
feature_learning_optimizer = optim.SGD(list(feature_learner_first.parameters()) + list(feature_learner_second.parameters()), 0.001, 0.9, 0.1)

#optimise the fine-tuning layer
fine_tuning_optimizer = optim.SGD(fine_tuning_head.parameters(), 0.0001, 0.99, 0.1)

#Our Cross-Entropy loss
cross_entropy_loss = nn.CrossEntropyLoss()
#cosine_sim = lambda x,  y: nn.NLLLoss()(nn.CosineSimilarity(dim=-1)(x,y))

# Train the encoder
num_epoch = 3
num_pre_train_head = 1000
num_train_encoder = 10
write_interval = 10

for current_epoch in range(num_epoch):
    for i, current_batch in enumerate(loader):
        element = rebatch_with_labels(current_batch).to(device)

        #Pre-train the head
        set_requires_grad(encoder, False)
        for j in range(num_pre_train_head):
            visual_head_optimizer.zero_grad()
            twod_features = feature_learner_first(element)

            logits = feature_learner_second(twod_features) #gives cosine similarity
            loss = cross_entropy_loss(logits, labels_index)
            #loss = cosine_sim(logits, labels_one_hot)
            run_avg.add(loss)
            if j%write_interval ==0:
                writer.add_scalar("loss_head", run_avg.get(), j)
            loss.backward()
            visual_head_optimizer.step()
            #feature_learner_second.re_norm_weights()

        writer.close()
        points = np.transpose(twod_features.cpu().detach().numpy())
        weight_vectors = feature_learner_second.m.weight.data.cpu().numpy().transpose()
        weight_vectors = weight_vectors / np.sqrt(weight_vectors[0,0]**2+ weight_vectors[1,0]**2)
        fig = plot_by_categories(points[0], points[1], labels_index.cpu().numpy(), weight_vectors, batch_size,
                                 batch_size * num_augments)

        plt.show()

        #Train the encoder
        run_avg.wipe()
        set_requires_grad(encoder, True)
        for j in range(num_train_encoder):
            feature_learning_optimizer.zero_grad()
            twod_features = feature_learner_first(element)
            logits = feature_learner_second(twod_features)
            loss = cross_entropy_loss(logits, labels_index)
            run_avg.add(float(loss))
            loss.backward()
            feature_learning_optimizer.step()

        writer.add_scalar('Loss/train', run_avg.get(), )






    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].imshow(element[0][0])
    # axarr[0, 1].imshow(element[1][0])
    # axarr[1, 0].imshow(element[2][0])
    # axarr[1, 1].imshow(element[3][0])
    # plt.show()
    # print(labels[0])
    # print(labels[1])
    # print(labels[2])
    # print(labels[3])
    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].imshow(element[4][0])
    # axarr[0, 1].imshow(element[5][0])
    # axarr[1, 0].imshow(element[6][0])
    # axarr[1, 1].imshow(element[7][0])
    # plt.show()
    # print(labels[4])
    # print(labels[5])
    # print(labels[6])
    # print(labels[7])
    # print("end first batch")

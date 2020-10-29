import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

import os
import random
sys_random = random.SystemRandom()
import numpy as np


class AgDetDataset(Dataset):
    
    def __init__(self, root, domains, batchsz, k_shot, k_query, seed=0, mode='train'):
        self.domains = domains
        self.k_shot = k_shot
        self.k_query = k_query
        self.batchsz = batchsz

        pos_dirs = [os.path.join(root, 'train/positive/2015.11Soergels', p) for p in os.listdir(os.path.join(root, 'train/positive/2015.11Soergels'))]
        neg_dirs = [os.path.join(root, 'train/negative/2015.11Soergels', p) for p in os.listdir(os.path.join(root, 'train/negative/2015.11Soergels'))]
        pos_dirs += [os.path.join(root, 'val/positive/2015.11Soergels', p) for p in os.listdir(os.path.join(root, 'val/positive/2015.11Soergels'))]
        neg_dirs += [os.path.join(root, 'val/negative/2015.11Soergels', p) for p in os.listdir(os.path.join(root, 'val/negative/2015.11Soergels'))]

        if mode == 'train':
            random.seed(seed)
            random.shuffle(pos_dirs)
            pos_dirs = pos_dirs[:int(1.0 * len(pos_dirs))]
            random.seed(seed)
            random.shuffle(neg_dirs)
            neg_dirs = neg_dirs[:int(1.0 * len(neg_dirs))]
        else:
            random.seed(seed)
            random.shuffle(pos_dirs)
            pos_dirs = pos_dirs[int(0.75 * len(pos_dirs)):]
            random.seed(seed)
            random.shuffle(neg_dirs)
            neg_dirs = neg_dirs[int(0.75 * len(neg_dirs)):]

        domain2dirs = dict([(d, [p for p in pos_dirs + neg_dirs \
                                    if d + '_' in p]) for d in domains])


        im2label = {}
        domain2ims = {}
        for k in domain2dirs.keys():
            dirs = domain2dirs[k]
            examples = []
            labels = []
            for p in dirs:
                examples += [os.path.join(root, p, "Images", e) for e in os.listdir(os.path.join(root, p, "Images"))]
                label = int("positive" in p)
                labels += [label for i in range(len(examples))]
            random.seed(seed)
            random.shuffle(examples)
            random.seed(seed)
            random.shuffle(labels)
            domain2ims[k] = examples
            for (e, l) in zip(examples, labels):
                im2label[e] = l

        self.domain2ims = domain2ims
        self.im2label = im2label

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize(224),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.support_x_batch, self.support_y_batch, self.query_x_batch, self.query_y_batch = None, None, None, None
        self.create_batch(self.batchsz)

    def create_batch(self, batchsz):
        support_x_batch = []
        support_y_batch = []
        query_x_batch = []
        query_y_batch = []

        for b in range(self.batchsz):
            domain = np.random.choice(list(self.domain2ims.keys()), 1, False)[0]
            ims = self.domain2ims[domain]
            indices = np.random.choice(len(ims), self.k_shot + self.k_query, False)
            ims = np.array(ims)[indices]
            labels = [self.im2label[im] for im in ims]
            if sys_random.choice([0, 1]):
                labels = [int(not l) for l in labels]
            support_x_batch.append(ims[:self.k_shot])
            support_y_batch.append(labels[:self.k_shot])
            query_x_batch.append(ims[self.k_shot:])
            query_y_batch.append(labels[self.k_shot:])

        self.support_x_batch = support_x_batch
        self.support_y_batch = support_y_batch
        self.query_x_batch = query_x_batch
        self.query_y_batch = query_y_batch
            

    def __len__(self):
        return self.batchsz
    
    def __getitem__(self, idx):
        support_x = self.support_x_batch[idx]
        support_y = self.support_y_batch[idx]
        query_x = self.query_x_batch[idx]
        query_y = self.query_y_batch[idx]

        support_x = [self.transform(x) for x in support_x]
        query_x = [self.transform(x) for x in query_x]

        return torch.stack(support_x), torch.Tensor(support_y).long(), torch.stack(query_x), torch.Tensor(query_y).long()

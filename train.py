import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from net import Network
from skimage.io import imsave



num_epoch = 1000
validation_split = 0.2
mu = 1.1
samples_dir = 'samples/'
transformations = transforms.Compose([
    #transforms.Resize((32,32)),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #transforms.Resize((32, 32))
])

def transform_image(img):
    transformations = transforms.Compose([
        #transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Resize((32, 32))
    ])
    img = transformations(img)

    return img

dataset = datasets.ImageFolder("data/celebA", transform = transformations)

dataset_size = len(dataset)
print(dataset_size)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=valid_sampler)
model = Network()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
criterion = nn.CrossEntropyLoss()

train_iter = iter(train_loader)
w_file = open("log.txt", "w")
w_file.write("Epoch\tLoss\n")
for e in range(num_epoch):
    try:
        inputs, _ = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        inputs, _ = next(train_iter)
    
    #inputs = inputs.to(device)
    lr_batch = torch.zeros(32, 3, 8, 8, dtype=torch.float)
    hr_batch = torch.zeros(32, 3, 32, 32, dtype=torch.float)
    labels = torch.zeros(32, 3, 32, 32, dtype = torch.float)

    for index, img in enumerate(inputs):
        img = transforms.ToPILImage()(img)
        hr = transforms.Resize((32,32))(img)
        labels[index] = transforms.ToTensor()(hr) * 255
        lr = transforms.Resize((8,8))(img)
        hr_batch[index] = transform_image(hr)
        lr_batch[index] = transform_image(lr)
    #print(hr_batch)
    #print(list(labels.getdata()))
    #print(lr_batch)
    #break
    #labels = np.array(labels)
    #print(labels.shape)

    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)
    labels = labels.to(device)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
        return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)

    def logits_to_pixel_value(logits, mu = 1.1):
        logits = logits.data.cpu().numpy()
        rebalance_logits = logits * mu
        probs = softmax(rebalance_logits)
        pixel_dict = np.arange(0, 256, dtype=np.float32)
        pixels = np.sum(probs * pixel_dict, axis=1)
        return np.floor(pixels)

    def save_samples(np_imgs, img_path):
        """
        Args:
            np_imgs: [N, H, W, 3] float32
            img_path: str
        """
        np_imgs = np_imgs.astype(np.uint8)
        N, H, W, _ = np_imgs.shape
        num = int(N ** (0.5))
        merge_img = np.zeros((num * H, num * W, 3), dtype=np.uint8)
        for i in range(num):
            for j in range(num):
                merge_img[i*H:(i+1)*H, j*W:(j+1)*W, :] = np_imgs[i*num+j,:,:,:]
        imsave(img_path, merge_img)

    if e % 5 == 1:
        gen_hr_imgs = torch.zeros(32, 3, 32, 32, dtype=torch.float)
        gen_hr_imgs = gen_hr_imgs.to(device)
        np_c_logits = model.conditioning_logits(lr_batch)
        for i in range(32):
            for j in range(32):
                for c in range(3):
                    np_p_logits = model.prior_logits(gen_hr_imgs)
                    new_pixel = logits_to_pixel_value(np_c_logits[:, c*256:(c+1)*256, i, j] + np_p_logits[:, c*256:(c+1)*256, i, j], mu=1.1)
                    gen_hr_imgs[:, c, i, j] = torch.tensor(new_pixel)
        gen_hr_imgs = gen_hr_imgs.data.cpu().numpy()
        gen_hr_imgs = np.transpose(gen_hr_imgs, (0, 3, 1, 2))
        save_samples(lr_batch.data.cpu().numpy(), samples_dir + '/lr_' + str(mu*10) + '_' + str(e) + '.jpg')
        save_samples(hr_batch.data.cpu().numpy(), samples_dir + '/hr_' + str(mu*10) + '_' + str(e) + '.jpg')
        save_samples(gen_hr_imgs, samples_dir + '/generate_' + str(mu*10) + '_' + str(e) + '.jpg')
        continue


    
    prior_logits, conditioning_logits = model.forward(hr_batch, lr_batch)

    optimizer.zero_grad()

    def calc_losses(logits, labels):
        logits = logits.reshape(-1, 256)
        labels = torch.tensor(labels, dtype = torch.int64)
        labels = labels.reshape(-1)
        labels = labels.to(device)
        return criterion(logits, labels)
    
    loss1 = calc_losses(prior_logits + conditioning_logits, labels)
    loss2 = calc_losses(conditioning_logits, labels)
    loss3 = calc_losses(prior_logits, labels)

    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    print("Epoch: {}  Loss: {}".format(e+1, loss))
    w_file.write(str(e+1)+"\t"+str(loss.data[0])+"\n")
    w_file.close()
    

    





import shutil 
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from torch.utils import data
import torch
import matplotlib.pyplot as plt
import random

def separate_data(dataset_photo, dataset_style):
    data_dir_photo = dataset_photo
    data_dir_style = dataset_style
    train_dir = 'train'
    test_dir = 'test'

    class_name = 'photo'
    for dir_name in [train_dir, test_dir]:
        os.makedirs(os.path.join(class_name, dir_name), exist_ok=True)

    source_dir = os.path.join(data_dir_photo)
    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
        if i % 10 != 0:
            dest_dir = os.path.join(class_name, train_dir) 
    #     elif i >10:
    #         break
        else:
            dest_dir = os.path.join(class_name, test_dir)
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
        
    class_name = 'style'
    for dir_name in [train_dir, test_dir]:
        os.makedirs(os.path.join(class_name, dir_name), exist_ok=True)

    source_dir = os.path.join(data_dir_style)
    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
        if i % 10 != 0:
            dest_dir = os.path.join(class_name, train_dir) 
    #     elif i >10:
    #         break
        else:
            dest_dir = os.path.join(class_name, test_dir)
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


class Mydataset(data.Dataset):
    def __init__(self,root,transform=None):   #инициализация
        imgs=os.listdir(root)
        self.imgs=sorted([os.path.join(root,img) for img in imgs])  #берем нужные изображения
        self.transform=transform   #преобразуем
        
    def __getitem__(self,index):   #Generates one sample of data
        img_path=self.imgs[index]
        image=Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):   #total number of samples
        return len(self.imgs)

class ImagePool():  #создаем пул изображений уже созданных дискриминатором, чтобы он мог учитывать ошибки
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # создаем пустой пул
            self.num_imgs = 0
            self.images = []

    def query(self, images): #возвращаем пул изображений
        if self.pool_size == 0:  # если задали 0, прошлые изображения не учитываются
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0) #новый тензор размером [image.data] , 4d tensor (BxNxHxW)
            if self.num_imgs < self.pool_size:   # пока не заполним пул, добавляем изображения
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # с вероятностью 50% в пул пойдет ранее сгенерированное изображение, а текущее сохраним
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # с вероятностью 50% в пул пойдет текущее изображение
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # добавляем выбранное изображение в пул
        return return_images

def to_np(x):
    return x.data.cpu().numpy()
def plot_train_result(real_image, gen_image, recon_image, epoch, save=False,  show=True, fig_size=(15, 15)):
    fig, axes = plt.subplots(2, 3, figsize=fig_size)
    imgs = [to_np(real_image[0]), to_np(gen_image[0]), to_np(recon_image[0]),
            to_np(real_image[1]), to_np(gen_image[1]), to_np(recon_image[1])]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        img = img.squeeze()
        img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    title = 'Epoch {0}'.format(epoch + 1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        save_fn = 'images/Result_epoch_{:d}'.format(epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()
        
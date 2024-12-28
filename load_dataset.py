import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'unm of datasets:{len(ds_train)}')

image,target=ds_train[1]
print(type(image),target)

plt.imshow(image,cmap='gray_r',vmin=0,vmax=255)
plt.title(target)
plt.show()

image_tensor = transforms.ToTensor()(image) 
image_tensor = image_tensor.to(dtype=torch.float32)  
print(image_tensor.shape, image_tensor.dtype)
print(image_tensor.min().item(), image_tensor.max().item())

#for i in range(5):
 #   for j in range(5):
  #   plt.subplot(5,5,k+1)
     #   plt.imshow(image,cmap='gray_r')

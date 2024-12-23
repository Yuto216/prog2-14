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

plt.imshow(image,cmap='gray_r')
plt.title(target)
plt.show()

image_tensor=transforms.functional.to_image(image)
image_tensor=transforms.functional.to_dtype(image,dtype=torch.float32,scale=True)
print(image_tensor.shape,image_tensor.dtype)
print(image.min(),image.max())

#for i in range(5):
 #   for j in range(5):
  #   plt.subplot(5,5,k+1)
     #   plt.imshow(image,cmap='gray_r')

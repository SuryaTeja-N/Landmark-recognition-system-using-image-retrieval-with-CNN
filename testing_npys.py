import numpy as np
images = np.load('images.npy')
labels = np.load('labels.npy')

print(f'Images shape: {images.shape}')
print(f'Labels shape: {labels.shape}')
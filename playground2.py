

import numpy as np

import torch.nn.functional as F
from torchgeometry.losses.one_hot import one_hot

import torch
import cv2



a = np.array([[[[1.2, 1.4, 1.9],[1.5, 1.9, 0.4],[0.2, 0.1, 12]],[[1.2, 1.4, 16],[1.5, 4, 0.4],[0.2, 0.1, 0.12]],[[12, 1.4, 1.6],[1.5, 1.7, 0.4],[0.2, 0.1, 12]]]])
#a = np.uint8(255*np.array([[[[0, 1, 0],[0, 1, 0],[0, 1, 0]],[[1, 0, 0],[1, 0, 0],[1, 0, 0]],[[0, 0, 1],[0, 0, 1],[0, 0, 1]]]]))
##print(a)
#cv2.imshow('a', a[0,:,:,:])
#cv2.waitKey()
#cv2.destroyAllWindows()

#a = np.array([[[0.0, 1.0, 2.0]]])
t = torch.from_numpy(a)
t = F.softmax(t, dim=3)

#print(t)
#print(t.shape)



#print(t.shape)

t = one_hot(t, 3)

#print(t)




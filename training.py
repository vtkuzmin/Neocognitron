# -*- coding: utf-8 -*-
import numpy as np
from ClassNeocognitron import Neocognitron
from patterns import S_4_train
import matplotlib.pyplot as plt

model = Neocognitron()
model.train(np.array([10e4]*4))

#model.predict(S_4_train[0,2])

#for i in range(model.C4.shape[0]):
#    plt.matshow(model.C4[i,:,:]) 
#    plt.show()
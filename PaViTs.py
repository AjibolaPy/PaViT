#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import pydot_ng as pydot
import numpy as np
import tensorflow as tf
#import pydot
import graphviz
from tensorflow.image import extract_patches


# In[3]:


class config:
    patch_size=32
    img_shape=224
    num_patches=224//32
    


# In[ ]:





# In[4]:


def extract_patch(img, patch_size=config.patch_size, img_shape=config.img_shape,
                    strides=config.patch_size):
    strides
    config.patch_size=patch_size
    config.img_shape=img_shape
    assert img_shape%patch_size==0, 'image shape should be divisible by num of patches'
    patches=extract_patches(img, sizes=[1, config.patch_size, config.patch_size, 1], strides=[1, config.patch_size, config.patch_size, 1],
                       rates=[1, 1, 1, 1], padding='VALID')
    return patches


# In[ ]:





# In[5]:


class encoder(Layer):
    def __init__(self, dim, pos_embedding=True):
        super(encoder).__init__()
        self.dim=dim
        self.pos_embedding=pos_embedding
        self.linear_proj=Dense(dim)
        self.embed=Embedding(config.num_patches*config.num_patches, dim) 
        
    def __call__(self, x):
        
        x1=self.linear_proj(x)
        if self.pos_embedding:
            pos=tf.range(0, x.shape[1])
            x2=self.embed(pos)
            return x1+x2
        else:
            return x1
    

def MLP(x, dim):
    x=GlobalAveragePooling1D()(x)
    for i in range(8): #7
        x=Dense(dim, activation='relu')(x)
        x=Dense(dim, activation='relu')(x) 
    return x

    
    


# In[6]:


class PaViT:
    def __init__(self, out=19, activation='softmax', dim=126, pos_embedding=True, num_heads=12, dropout:bool=True):
        self.dropout=dropout
        self.out=out
        self.activation=activation
        self.dim=dim
        self.encoder=encoder(dim)
        self.attn=MultiHeadAttention(num_heads=num_heads,key_dim=dim)
        self.pos_embedding=pos_embedding
        
        
    def load_model(self):
        ins=Input(shape=(config.img_shape, config.img_shape, 3))
        patch=extract_patch(ins, patch_size=config.patch_size, img_shape=config.img_shape,
                    strides=config.patch_size)

        reshape=Reshape((-1, patch.shape[-1]))(patch)
        encode=encoder(dim=self.dim)(reshape)
        x=BatchNormalization()(encode)

        atn=self.attn(x, x)
        mlp=MLP(x, self.dim)
        add=Add()([atn, mlp])
        norm=BatchNormalization()(add)
        if self.dropout:
            norm=Dropout(.3)(norm)
        flat=Flatten()(norm)
        out=self.head(flat)

        
        return Model(ins, out)
    def head(self, x):
        linear=Dense(self.out,activation=self.activation)(x)
        return linear
model=PaViT().load_model()
model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





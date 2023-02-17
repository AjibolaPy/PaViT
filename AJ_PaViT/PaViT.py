#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
from tensorflow.keras.models import *
from keras.optimizers import Adam
import cv2
import tensorflow as tf
from keras.callbacks import *
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import matplotlib.pyplot as plt
from keras.activations import swish
from keras.preprocessing.image import *
from tensorflow.image import extract_patches

num_patches=224//3

class patches(Layer):
    def __init__(self,patch_size ):
        self.patch_size=patch_size
    def __call__(self, x):
        assert x.shape[1]%self.patch_size==0, 'Patch_size should be divisible'
        if len(list(tf.shape(x)))==2:
            x=tf.expand_dims(x, axis=-1)
        if len(list(tf.shape(x)))==3:
            x=tf.expand_dims(x, axis=0)
        patch=extract_patches(images=x,strides=[1, self.patch_size, self.patch_size, 1] ,sizes=[1, self.patch_size, self.patch_size, 1],rates=[1, 1, 1,1], padding='VALID')
        return patch
    
def encoder(x, dim=32,pos:bool=True):
    lin_proj=Dense(dim, activation='relu')
    if pos:
        pos_Emb=Embedding(x.shape[1], dim)
        position=tf.range(0, x.shape[1])
        return lin_proj(x)+pos_Emb(position)
    else:
        return lin_proj(x)
    
def Mlp(x, n:int=8, dim=32):
    x=GlobalAveragePooling1D()(x)
    for i in range(n): #7
        x=Dense(dim, activation='relu')(x)
        x=Dense(dim, activation='relu')(x)   
    
    return x



class PaViT:
    def __init__(self, shape=(224, 224, 3),num_heads=12, patch_size=32, dim=126, pos_emb:bool =False, 
                 mlp_it=8, attn_drop:int= .3, dropout:bool=True):
        self.dropout=dropout
        self.shape=shape
        self.num_heads=num_heads
        self.patch_size=patch_size
        self.dim=dim
        self.attn_drop=attn_drop
        self.pos_emb=pos_emb
        self.mlp_it=mlp_it
        
    def model(self, output_class=None, output=15, activation='softmax'):
        inp=Input(shape=self.shape, name='Input')
        patch=patches(patch_size=self.patch_size)(inp)
        reshape=Reshape((-1, patch.shape[-1]))(patch)
        encode=encoder(reshape, dim=self.dim, pos=True)
        x=BatchNormalization()(encode)
        drop=None
        if self.attn_drop:
            drop=self.attn_drop
        attn=MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim, dropout=drop)(x,x)   #12
        mlp=Mlp(x,n=self.mlp_it, dim=self.dim)
        add=Add()([mlp, attn])
        norm=BatchNormalization()(add)
        if self.dropout:
            norm=Dropout(.3)(norm)
        
        flat=Flatten()(norm)
        if not output_class:
            out=Dense(output, activation=activation)(flat)
        else: 
            out=output_class(flat)
        
        
        self.without_head=Model(inp, norm)
        return Model(inp, out)
    
    
    def remove_head(self):
        try:
            return self.without_head 
        except: 
            print('Cant load model without last layer. \nInitialize model first')
    

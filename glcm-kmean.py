import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data


def main():
    pass


def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=1):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    print(glcm)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    max_  = np.max(glcm, axis=(0,1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent




# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:20:11 2020

@author: ASH
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.Image as image
from sklearn.cluster import KMeans
def show(img):
    plt.imshow(img,plt.cm.gray)
    plt.show()
    return img


def load_data(result):
    data = []
    n,m= result.shape #活的图片大小
    imap=np.zeros((m,n))
    black=0
    for i in range(m):
        for j in range(n):  #将每个像素点RGB颜色处理到0-1范围内并存放data
            point=result[j][i]
            if not point==black:
                x = result[j][i]
                #y= result[j][i][1]
                #z= result[j][i][2]
                imap[i][j]=1
                data.append([x/255.0])
    return data,m,n,imap #以矩阵型式返回data，图片大小
file = r'ash.jpg'
def kmean(img):
    #img = cv2.imread(file,cv2.IMREAD_COLOR)  
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    plt.imshow(img)
    img_data,row,col,imap = load_data(img)
    label = KMeans(n_clusters=2).fit_predict(img_data)  #聚类中心的个数为2

    pic_new = image.new("P",(row,col))  #创建一张新的灰度图保存聚类后的结果
    index=-1
    for j,row in enumerate(imap):
        for i,point in enumerate(row):
            if  point==0.:
                pic_new.putpixel((j,i),255)
            else:
                index+=1
                pic_new.putpixel((j,i),int(100*label[index]))
    img = np.array(pic_new)
    plt.imshow(img,plt.cm.rainbow)            
    a=0
    b=0
    for row in img:
        for point in row:
            if point!=255:
                if point ==0:
                    a+=1
                elif point==100:
                    b+=1
    print(a,b,a/b,b/a)#不确定分子分母顺序倒没倒哦



import numpy as np
from skimage import data
from matplotlib import pyplot as plt
from PIL import Image

def main():
    pass


if __name__ == '__main__':
    path = r"ash.jpg"#读取路径
    img=np.array(Image.open(path).convert('L')) #灰度转换
    mean = fast_glcm_mean(img)   #glcm 计算
    kmean(mean)               #kmean


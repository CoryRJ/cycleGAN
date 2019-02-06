import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt

class Batch:
    def __init__(self,num_of_img,path,img_size=(250,250,3),batch_size=1,start=0,img_type='.png'):
        self.num_of_img = num_of_img
        self.path = path
        self.batch_size= batch_size
        self.start = start
        self.img_type = img_type
        self.img_size = img_size
        self.digits = len(str(num_of_img))
        self.pic_index=self.start

    def read_pic(self,id):
        name = ''
        for _ in range(self.digits-len(str(id))):
            name = name +'0'
        name = self.path + name + str(id) + self.img_type
        img = Image.open(name)
        return np.asarray(img)[:,:,0:3]

    def get_batch(self):
        pictures = np.empty([self.batch_size,self.img_size[0],self.img_size[1],self.img_size[2]])
        for ind in range(self.batch_size):
            pictures[ind] = self.read_pic(self.pic_index)
            self.pic_index = self.pic_index + 1
            if(self.pic_index > self.num_of_img):
                self.pic_index = self.start
        return pictures/256.0

    def get_batch_random(self,num=None):
        pictures = None
        grab_size = None
        if(num == None):
            pictures = np.empty([self.batch_size,self.img_size[0],self.img_size[1],self.img_size[2]])
            grab_size = self.batch_size
        else:
            pictures = np.empty([num,self.img_size[0],self.img_size[1],self.img_size[2]])
            grab_size = num
        for ind in range(grab_size):
            pictures[ind] = self.read_pic(randint(self.start,self.num_of_img))
        return pictures/256.0

if __name__ == '__main__':
    num_of_img = 9483
    path = 'C:\\Users\\redhe\\python_stuff\\yn_frames\\pictures\\'
    img_size = (168,300,3)
    batch_size=5
    start=0
    img_type ='.png'
    test = Batch(num_of_img,path,img_size,batch_size,start,img_type)
    pic = test.get_batch_random()
    #m = test.read_pic(pic[i])
    for i in range(batch_size):
        plt.figure('Testing:')
        plt.imshow(pic[i])
        plt.show()


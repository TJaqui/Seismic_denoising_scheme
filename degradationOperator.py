# This is a sample Python script.
import torch
from degradeFunctions import degradeImage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import time

def degradeBatch(img):


    dm_tp = np.arange(0,16)
    np.random.shuffle(dm_tp)


    damages = []
    clean = []
    cl_img = img.clone()
    cl_img = cl_img.float()
    cl_img -= cl_img.min()
    cl_img /= cl_img.max()
    #damages.append(cl_img)
    for i in range(3):
        dm_img = img.clone().float()
        dm_img -= dm_img.min()
        dm_img /= dm_img.max()
        typeNoise = dm_tp[i]

        if (typeNoise == 0):

            sigma = np.random.random()
            intensity = np.random.uniform(0.35, 0.45)
            dm_img = degradeImage(dm_img).gaussianNoise(sigma, intensity)


        elif (typeNoise == 1):

            intensity = np.random.uniform(0.3, 0.65)
            dm_img = degradeImage(dm_img).poissonNoise(intensity)


        elif (typeNoise == 2):

            intensity = np.random.uniform(0.2, 0.3)
            frec = np.random.randint(10, 20)
            dm_img = degradeImage(dm_img).stripesNoise(intensity, frec)

        elif (typeNoise == 3):

            amount = np.random.uniform(0.04, 0.15)
            dm_img = degradeImage(dm_img).impulseNoise(amount)

        elif (typeNoise == 4):

            intensity = np.random.uniform(0.15, 0.3)
            dm_img = degradeImage(dm_img).speckleNoise(intensity)


        elif (typeNoise == 5):

            intensity = np.random.uniform(0.2, 0.4)
            dm_img = degradeImage(dm_img).convolutionG1(intensity)


        elif (typeNoise == 6):

            intensity = np.random.uniform(0.2, 0.3)
            dm_img = degradeImage(dm_img).convolutionG2(intensity)

        elif (typeNoise == 7):

            intensity = np.random.uniform(0.2, 0.4)
            dm_img = degradeImage(dm_img).convolutionG1V(intensity)

        elif (typeNoise == 8):

            intensity = np.random.uniform(0.25, 0.4)
            amount = np.random.uniform(0.8, 1)
            dm_img = degradeImage(dm_img).convolutionG1I(intensity, amount)

        elif (typeNoise == 9):

            sigma = np.random.uniform(1.8, 2.2)
            dm_img = degradeImage(dm_img).gaussianBlur(sigma)

        elif (typeNoise == 10):

            amount = np.random.randint(15,40)
            dm_img = degradeImage(dm_img).streak(amount)

        elif (typeNoise == 11):

            amount = np.random.randint(15,40)
            #intensity = np.random.uniform(0.7, 3)
            dm_img = degradeImage(dm_img).lines(amount)

        elif (typeNoise == 12):

            overlap1 = np.random.randint(1, 40)
            overlap2 = np.random.randint(40,70)
            intensity = np.random.uniform(0.15, 0.35)
            dm_img = degradeImage(dm_img).waves(intensity, overlap1, overlap2)

        elif (typeNoise == 13):

            amp1 = np.random.uniform(0.002, 0.02)
            amp2 = np.random.uniform(0.002, 0.02)
            intensity = np.random.uniform(0.15, 0.3)
            sigma = np.random.uniform(0.8, 1.0)

            rot = np.random.randint(0,1)
            dm_img = degradeImage(dm_img).waves2(amp1, amp2, intensity, sigma, rot)

        elif (typeNoise == 14):

            intensityG2 = np.random.uniform(0.23, 0.3)
            intensityG1 = np.random.uniform(0.23, 0.4)
            dm_img = degradeImage(dm_img).s1(intensityG1, intensityG2)

        elif (typeNoise == 15):

            intensityG2 = np.random.uniform(0.25, 0.5)
            intensityG1 = np.random.uniform(0.25, 0.5)
            sigma=1
            dm_img = degradeImage(dm_img).s1Blur(intensityG1, intensityG2, sigma)


        clean.append(cl_img)

        damages.append(dm_img)
    clean.append(cl_img)
    damages.append(degradeImageMultiple(img))
    cln = torch.stack(clean, dim=0)
    dmgs = torch.stack(damages, dim=0)
    return dmgs, cln

def degradeImageMultiple(img):
    """
        The function degrade each clean image randomly with 3 kinds of damage.
        :param img: The "img" parameter refers to the images that will be damaged
        :return: The function return a degraded image
        """
    dm_tp = np.arange(0, 16)
    np.random.shuffle(dm_tp)

    dm_img = img.clone()

    for i in range(3):
        typeNoise = dm_tp[i]

        if (typeNoise == 0):

            sigma = np.random.random()
            intensity = np.random.uniform(0.3, 0.4)
            dm_img = degradeImage(dm_img).gaussianNoise(sigma, intensity)


        elif (typeNoise == 1):

            intensity = np.random.uniform(0.3, 0.65)
            dm_img = degradeImage(dm_img).poissonNoise(intensity)


        elif (typeNoise == 2):

            intensity = np.random.uniform(0.15, 0.3)
            frec = np.random.randint(10, 20)
            dm_img = degradeImage(dm_img).stripesNoise(intensity, frec)

        elif (typeNoise == 3):

            amount = np.random.uniform(0.04, 0.15)
            dm_img = degradeImage(dm_img).impulseNoise(amount)

        elif (typeNoise == 4):

            intensity = np.random.uniform(0.1, 0.3)
            dm_img = degradeImage(dm_img).speckleNoise(intensity)


        elif (typeNoise == 5):

            intensity = np.random.uniform(0.15, 0.4)
            dm_img = degradeImage(dm_img).convolutionG1(intensity)


        elif (typeNoise == 6):

            intensity = np.random.uniform(0.15, 0.3)
            dm_img = degradeImage(dm_img).convolutionG2(intensity)

        elif (typeNoise == 7):

            intensity = np.random.uniform(0.15, 0.4)
            dm_img = degradeImage(dm_img).convolutionG1V(intensity)

        elif (typeNoise == 8):

            intensity = np.random.uniform(0.2, 0.4)
            amount = np.random.uniform(0.8, 1)
            dm_img = degradeImage(dm_img).convolutionG1I(intensity, amount)

        elif (typeNoise == 9):

            sigma = np.random.uniform(1.7, 2.2)
            dm_img = degradeImage(dm_img).gaussianBlur(sigma)

        elif (typeNoise == 10):

            amount = np.random.randint(10,50)
            dm_img = degradeImage(dm_img).streak(amount)

        elif (typeNoise == 11):

            amount = np.random.randint(10,50)
            intensity = np.random.uniform(0.7, 3)
            dm_img = degradeImage(dm_img).lines(amount)

        elif (typeNoise == 12):

            overlap1 = np.random.randint(10, 40)
            overlap2 = np.random.randint(40,70)
            intensity = np.random.uniform(0.1, 0.35)
            dm_img = degradeImage(dm_img).waves(intensity, overlap1, overlap2)

        elif (typeNoise == 13):

            amp1 = np.random.uniform(0.001, 0.02)
            amp2 = np.random.uniform(0.001, 0.02)
            intensity = np.random.uniform(0.1, 0.3)
            sigma = 1
            rot = np.random.randint(0,1)
            dm_img = degradeImage(dm_img).waves2(amp1, amp2, intensity, sigma, rot)

        elif (typeNoise == 14):

            intensityG2 = np.random.uniform(0.15, 0.3)
            intensityG1 = np.random.uniform(0.15, 0.4)
            dm_img = degradeImage(dm_img).s1(intensityG1, intensityG2)

        elif (typeNoise == 15):

            intensityG2 = np.random.uniform(0.25, 0.5)
            intensityG1 = np.random.uniform(0.25, 0.7)
            sigma=1
            dm_img = degradeImage(dm_img).s1Blur(intensityG1, intensityG2, sigma)

    return dm_img

def degradeBatchMatrix(batch):
    """
    The function degrade images and return a matrix. The first position of every row is the original image,
    and each of the following positions represents a type of degradation that affects the original image
    :param batch: The "batch" parameter refers to the array of images that will be damaged
    :return: The function return a matrix with degradations
    """
    degradedBatch = np.zeros((len(batch), 13), dtype=object)
    print(1)
    for i in range(degradedBatch.shape[0]):
        degradedBatch[i][0] = batch[i]
        degradedBatch[i][1] = degradeImage(batch[i]).impulseNoise(30)
        degradedBatch[i][2] = degradeImage(batch[i]).gaussianNoise(1)
        degradedBatch[i][3] = degradeImage(batch[i]).poissonNoise(0.3)
        degradedBatch[i][4] = degradeImage(batch[i]).speckleNoise(0.3)
        degradedBatch[i][5] = degradeImage(batch[i]).convolutionG1(0.4)
        degradedBatch[i][6] = degradeImage(batch[i]).convolutionG2(0.6)
        degradedBatch[i][7] = degradeImage(batch[i]).convolutionG1V(0.5)
        degradedBatch[i][8] = degradeImage(batch[i]).convolutionG1I(0.4, 30)
        degradedBatch[i][9] = degradeImage(batch[i]).gaussianBlur(1)
        degradedBatch[i][10] = degradeImage(degradedBatch[i][5]).convolutionG2(0.4)
        degradedBatch[i][11] = degradeImage(degradedBatch[i][10]).gaussianBlur(1)
        degradedBatch[i][12] = degradeImage(degradedBatch[i][11]).convolutionG1I(0.3,20)
    return degradedBatch


'''
if __name__ == '__main__':
  
    imge=[]
    x1 = []
    data1 = Image.open(f'../data/patch_26.jpg')
    data2 = Image.open(f'../data/patch_29.jpg')
    data1 = np.array(data1)
    data2 = np.array(data2)
    data1 = torch.from_numpy(data1)
    data2 = torch.from_numpy(data2)
    #imge.append(data1)
    #imge.append(data2)
    #for i in imge:
     #   x1.append(degradeBatch(i))
    start = time.time()
    s = degradeImage(data1).speckleNoise(0.35)
    end = time.time()
    print(end-start)
    #all = torch.stack(x1,dim=0)
       
    print(all.shape)
    for i in range(all.shape[0]):
        print(all[i].shape)
        for j in range(all.shape[1]):
            print(all[i][j].shape)
    
    imge=[]
    x1 = []
    start = time.time()
    for i in glob.glob('C:/Users/juter/Desktop/Atenuación de ruido/data/*.jpg'):
        data = Image.open(i)
        data = np.array(data)
        data = torch.from_numpy(data)
        imge.append(data)
    end = time.time()
    print("loading time images: ", end-start)
    start = time.time()
    mn = []
    c=0

    for i in imge:
        start1 = time.time()
        x1.append(degradeBatch(i))
        end1 = time.time()
        mn.append(end1-start1)
        c += 1
        print("image number: ", c)
        print("degrade image time: ", end1-start1)
    end = time.time()
    print("degrading batch time: ", end-start)
    print("mean time image degradation: ", np.mean(mn))

    
fig = plt.figure(figsize=(6, 6), dpi=300)
fig.add_subplot(2, 2, 1)
plt.imshow(data1.numpy(), cmap="gray")
plt.axis("off")
plt.title("clean", fontsize=3)
fig.add_subplot(2, 2, 2)
plt.imshow(s.numpy(), cmap="gray")
plt.axis("off")
plt.title("dmg", fontsize=3)
plt.show()


fig.add_subplot(4, 4, 3)
plt.imshow(x1[2].numpy(), cmap="gray")
plt.axis("off")
plt.title("clean", fontsize=3)
fig.add_subplot(4, 4, 4)
plt.imshow(x1[3].numpy(), cmap="gray")
plt.axis("off")
plt.title("dmg", fontsize=3)
fig.add_subplot(4, 4, 5)
plt.imshow(x1[4].numpy(), cmap="gray")
plt.axis("off")
plt.title("clean", fontsize=3)
fig.add_subplot(4, 4, 6)
plt.imshow(x1[5].numpy(), cmap="gray")
plt.axis("off")
plt.title("dmg", fontsize=3)
fig.add_subplot(4, 4, 7)
plt.imshow(x1[6].numpy(), cmap="gray")
plt.axis("off")
plt.title("clean", fontsize=3)
fig.add_subplot(4, 4, 8)
plt.imshow(x1[7].numpy(), cmap="gray")
plt.axis("off")
plt.title("dmg", fontsize=3)
fig.add_subplot(4, 4, 9)
plt.imshow(x1[8].numpy(), cmap="gray")
plt.axis("off")
plt.title("clean", fontsize=3)
fig.add_subplot(4, 4, 10)
plt.imshow(x1[9].numpy(), cmap="gray")
plt.axis("off")
plt.title("dmg", fontsize=3)

plt.show()




fig.add_subplot(4, 4, 5)
plt.imshow(x5, cmap="gray")
plt.axis("off")
plt.title("stripes", fontsize=3)
fig.add_subplot(4, 4, 6)
plt.imshow(x6, cmap="gray")
plt.axis("off")
plt.title("covg1", fontsize=3)
fig.add_subplot(4, 4, 7)
plt.imshow(x7, cmap="gray")
plt.axis("off")
plt.title("covg2", fontsize=3)
fig.add_subplot(4, 4, 8)
plt.imshow(x8, cmap="gray")
plt.axis("off")
plt.title("covg1v", fontsize=3)
fig.add_subplot(4, 4, 9)
plt.imshow(x9, cmap="gray")
plt.axis("off")
plt.title("blur", fontsize=3)
fig.add_subplot(4, 4, 10)
plt.imshow(x10, cmap="gray")
plt.axis("off")
plt.title("s1", fontsize=3)
fig.add_subplot(4, 4, 11)
plt.imshow(x11, cmap="gray")
plt.axis("off")
plt.title("s1blur", fontsize=3)
fig.add_subplot(4, 4, 12)
plt.imshow(x12, cmap="gray")
plt.axis("off")
plt.title("waves", fontsize=3)
fig.add_subplot(4, 4, 13)
plt.imshow(x13, cmap="gray")
plt.axis("off")
plt.title("waves2", fontsize=3)
plt.show()
'''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

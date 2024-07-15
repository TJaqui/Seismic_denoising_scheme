import numpy as np
import cv2
import scipy as sc
import skimage as ski
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
torch.pi = torch.Tensor([math.pi])
class degradeImage:

    def __init__(self, img):
        '''
        :param img: The 'img' parameter refers to the image to modify
        '''
        self.original = img
        self.img = img
        self.img = self.img.numpy()
        self.img = self.img.astype(dtype=np.float32)
        self.img -= self.img.min()
        self.img /= self.img.max()
        self.img = torch.from_numpy(self.img)

    def impulseNoise(self, AMOUNT):
        '''
        A matrix of zeros with the dimensions of image is generated.
        to make sure there are no issues, the 'dtype' is specified to be the same as 'img'
        The zero matrix is refilled with randomly generated values that follow a
        uniformly distributed function with values of 0 and 255
        To create the noise, a threshold is applied to filter the amount of points.
        The 'thres' value represents the maximum value that a pixel can have. If this value is exceeded,
        it is then changed based on the next parameter, which is fixed (the functionality of this threshold 'thres'
        was configured to be inversed, this means, the bigger the 'thres' value, the bigger the noise).
        Then the noise is added to the image.
        :param thres: The 'thres' value represents the maximum value that a pixel can have. If this value is exceeded, it is then changed based on the next parameter, which is fixed wit the value 0.1.
        :return: it is returned the image with impulse noise
        '''

        AMOUNT = AMOUNT
        SALT_VS_PEPPER = 0.6

        noisy_image_SP_tensor = self.img.clone()
        number_faulty_pixels = int(np.ceil(AMOUNT * self.img.numel()))
        indices = torch.randint(0, 128, (2, number_faulty_pixels))
        values = torch.tensor(np.random.binomial(1, SALT_VS_PEPPER, size=number_faulty_pixels), dtype=torch.float32)
        noisy_image_SP_tensor[indices[0], indices[1]] = values
        im_img = noisy_image_SP_tensor
        #noise = np.ones(self.img.shape) * 0.5
        #noise[indices[0], indices[1]] = values.numpy()
        '''
        imp_noise = np.zeros(self.img.shape, dtype=np.float32)
        cv2.randu(imp_noise, 0, 255)
        imp_noise = cv2.threshold(imp_noise, thres, 0.1, cv2.THRESH_BINARY_INV)[1]

        im_img = self.img + imp_noise
        '''
        im_img -= im_img.min()
        im_img /= im_img.max()

        return im_img

    def gaussianNoise(self,sigma, intensity):
        '''
        A matrix of zeros with the dimensions of image is generated.
        to make sure there are no issues, the 'dtype' is specified to be the same as image
        The zero matrix is now refilled with randomly generated values that follow the Gaussian distribution
        using the 'sigma' value to create the noise, Finally, the noise is added to image
        :param sigma: The 'sigma' parameter refers to the standard deviation used to generate the noise
        :return: it is returned the image with gaussian noise
        '''
        gauss_noise = torch.empty(self.img.shape).normal_(mean=0,std=sigma)
        gauss_noise -= gauss_noise.min()
        gauss_noise /= gauss_noise.max()
        #gauss_noise=torch.randn_like(self.img)
        gn_img = self.img + gauss_noise * intensity
        gn_img -= gn_img.min()
        gn_img /= gn_img.max()
        '''
        gauss_noise=np.zeros(self.img.shape,dtype=self.img.dtype)
        cv2.randn(gauss_noise,0,sigma)
        gauss_noise = (gauss_noise).astype(self.img.dtype)
        gauss_noise -= gauss_noise.min()
        gauss_noise /= gauss_noise.max()

        gn_img = self.img + gauss_noise
        gn_img -= gn_img.min()
        gn_img /= gn_img.max()
        '''
        return gn_img

    def speckleNoise(self, intensity):
        '''
        A matrix of random values with the dimensions of image is generated. Then the matrix is normalized.
        the image is multiplied by both the noise and the desired intensity,
        and the result is ultimately added to the image
        :param intensity: The 'intensity' parameter refers to the intensity of the noise

        :return: it is returned the image with speckle noise
        '''
        speck = torch.randn_like(self.img)
        #speck -=speck.min()
        #speck /= speck.max()

        '''
        row,col = self.img.shape
        speck = np.random.randn(row,col)
        speck -= speck.min()
        speck /= speck.max()
        '''
        sp_img = self.img + self.img * speck * intensity
        sp_img -= sp_img.min()
        sp_img /= sp_img.max()

        return sp_img

    def poissonNoise(self, intensity):
        '''
        Poisson noise is pixel-dependent, so a matrix with random values
        following the Poisson distribution is generated,
        and each pixel is condition by the previus value in the image
        The noise can be too strong, which is why we reduce its intensity
        by multiplying it by values between 0 and 1
        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :return: it is returned the image with speckle noise
        '''
        poisson_noise = torch.poisson(self.img)
        #poisson_noise = np.random.poisson(self.img)
        #poisson_noise = poisson_noise.astype(np.float64)
        poisson_noise -= poisson_noise.min()
        poisson_noise /= poisson_noise.max()

        poisson_noise = poisson_noise * intensity
        poi_img = self.img - poisson_noise
        poi_img -= poi_img.min()
        poi_img /= poi_img.max()

        return poi_img


    def stripesNoise(self, intensity, frec):
        '''
        It is taken the row, col values of the 'img'
        Using the values of row and col two 1D arrays are created with values
        between 0 and 1 with the two previous arrays a mesh is created with a 2D form
        The grid of X now is passed to the funtion to modelate the stripes,
        then the stripes multiplied by the desire intensity and then added to 'img'

        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :param frec: The 'frec' parameter refers to the frequency of the sin function
        :return: it is returned the image with stripes noise
        '''

        row, col = self.img.shape
        x = torch.linspace(0, 1, col)
        y = torch.linspace(0, 1, row)
        X, Y = torch.meshgrid(x, y)

        stripes = np.sin(frec * 2 * torch.pi * Y)
        #stripes -= stripes.min()
        #stripes /= stripes.max()

        st_img = self.img + stripes * self.img * intensity
        st_img -= st_img.min()
        st_img /= st_img.max()

        return st_img


    def lines(self, amount):
        '''
        To draw the lines, a list of arrays is initialized, starting with the first streak.
        To add the remaining streaks, a for loop is initiated, applying the formulas.
        The list is then converted into a NumPy array. OpenCV's 'polylines' function is used to draw the lines.
        The 'addWeighted' function it is applied to control the opacity of the lines.
        :param space: The 'space' parameter refers to the space betwen lines
        :param thickness: The 'thickness' parameter refers to the thickness of the lines
        :param pha: The 'pha' parameter refers to the alpha that controls the opacity of the image
        :return: it is returned the image with lines
        '''
        x = torch.randint(1, 128, (1, 1)).item()
        y = torch.randint(1, 128,(1,1)).item()
        streak_img = self.img.clone()


        val = streak_img[x][y]

        for i in range(amount):
            pos = np.random.randint(0, 128)
            for m in range(streak_img.shape[0]):
                streak_img[m][pos] = val
        ln_img = streak_img

        ln_img -= ln_img.min()
        ln_img /= ln_img.max()

        return ln_img

    def streak(self, amount):
        num = torch.randint(1, amount, (1,1)).item()

        streak_img = self.img
        vec = []

        for j in range(streak_img.shape[0]):
            vec.append(streak_img[j][num])

        for i in range(amount):
            pos = np.random.randint(0, 128)
            for m in range(streak_img.shape[0]):
                streak_img[m][pos] = vec[m]
        streak_img -= streak_img.min()
        streak_img /= streak_img.max()
        return streak_img

    def waves(self, intensity, overlap1, overlap2):

        waves_img = self.img

        hor, verti = waves_img.shape[0], waves_img.shape[1]
        a = torch.zeros(self.img.shape);

        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        [X, Y] = torch.meshgrid(x2, x1)

        r2 = torch.empty(1).uniform_(0.1, 0.7).item()
        r3 = torch.empty(1).uniform_(0.1, 0.7).item()
        r4 = torch.empty(1).uniform_(0.1, 0.7).item()

        for i, y in enumerate(x2, start=0):
            r = np.random.uniform(1, 1.7)
            for j, x in enumerate(x1, start=0):
                a[j, i] = torch.sin(x + (r * y ** 2))
        b = torch.sin(X + (r2 * Y ** 2))
        c = torch.sin(X + (r3 * Y ** 2))
        d = torch.sin(X + (r4 * Y ** 2))

        a -= a.min(); a /= a.max(); b -= b.min(); b /= b.max()
        c -= c.min(); c /= c.max(); d -= d.min(); d /= d.max()


        it = torch.randint(0,3,(1,1)).item()

        ra = torch.randint(overlap1, overlap2, (1,1)).item()
        ra1 = torch.randint(overlap1, overlap2, (1,1)).item()

        if it == 0:
            waves_img = waves_img + a * intensity * waves_img
        elif it == 1:
            waves_img = waves_img + b * intensity * waves_img
        else:
            z = b
            f = c
            s = d
            z[0:ra, 0:128] = f[0:ra, 0:128] + z[0:ra, 0:128]
            z[ra1:128, 0:128] = s[ra1:128, 0:128] + z[ra1:128, 0:128]
            z -= z.min()
            z /= z.max()
            waves_img = waves_img + z * intensity * waves_img

        waves_img -= waves_img.min()
        waves_img /= waves_img.max()

        return waves_img

    def waves2(self, amp1, amp2, intensity, sigma, rot):

        waves_img = self.img

        hor, verti = waves_img.shape[0], waves_img.shape[1]

        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        [X, Y] = torch.meshgrid(x2, x1)


        g2 = torch.sin(X + (((amp1) * (Y + 90) ** 2)))
        g3 = torch.sin(X + (((amp1) * (Y - 90) ** 2)))
        g7 = torch.sin(X + (((amp2) * (Y + 90) ** 2)))
        g8 = torch.sin(X - (((amp2) * (Y + 90) ** 2)))
        g4 = torch.sin(X + (((amp2) * (Y - 90) ** 2)))
        g5 = torch.sin(X - (((amp2) * (Y - 90) ** 2)))

        g2 -= g2.min(); g2 /= g2.max(); g3 -= g3.min(); g3 /= g3.max()
        g4 -= g4.min(); g4 /= g4.max(); g5 -= g5.min(); g5 /= g5.max()
        g7 -= g7.min(); g7 /= g7.max(); g8 -= g8.min(); g8 /= g8.max()
        it = torch.randint(0, 9, (1,1)).item()

        if rot == 0:
            if it == 0:
                waves_img = waves_img + (g2 + g3) * waves_img * intensity
            elif it == 1:
                waves_img = waves_img + (g2 + g4) * waves_img * intensity
            elif it == 2:
                waves_img = waves_img + (g2 + g5) * waves_img * intensity
            elif it == 3:
                waves_img = waves_img + (g2 + g7) * waves_img * intensity
            elif it == 4:
                waves_img = waves_img + (g2 + g8) * waves_img * intensity
            elif it == 5:
                waves_img = waves_img + (g3 + g4) * waves_img * intensity
            elif it == 6:
                waves_img = waves_img + (g3 + g5) * waves_img * intensity
            elif it == 7:
                waves_img = waves_img + (g3 + g7) * waves_img * intensity
            elif it == 8:
                waves_img = waves_img + (g3 + g8) * waves_img * intensity

        else:
            if it == 0:
                f = g2 + g3
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 1:
                f = g2 + g4
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 2:
                f = g2 + g5
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 3:
                f = g2 + g7
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 4:
                f = g2 + g8
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 5:
                f = g3 + g4
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 6:
                f = g3 + g5
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 7:
                f = g3 + g7
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity
            elif it == 8:
                f = g3 + g8
                f2 = np.rot90(f, 2)
                waves_img = waves_img + (f + f2) * waves_img * intensity

        waves_img = ski.filters.gaussian(
            waves_img, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
        waves_img = torch.from_numpy(waves_img)
        waves_img -= waves_img.min()
        waves_img /= waves_img.max()

        return waves_img


    def convolutionG1(self, intensity):
        '''
        The function generates correlated noise, with a convolution between a g funtion and a gaussian kernel.
        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :return: it is returned the image with a convolutional noise
        '''
        img = self.img
        hor, verti = img.shape[0], img.shape[1]
        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        X1, X2 = torch.meshgrid(x1, x2)


        g1 = (1 - torch.abs(X1)).clamp(min=0) * (16 - torch.abs(X2)).clamp(min=0)

        g1 = g1.unsqueeze(0).unsqueeze(0)
        gauss_noise = torch.randn(129,129).unsqueeze(0).unsqueeze(0)

        covG1 = F.conv2d(gauss_noise, g1, padding=63)
        covG1 = covG1.squeeze(0).squeeze(0)

        covG1 -= covG1.min()
        covG1 /= covG1.max()

        gn_covG1_img = img + covG1 * intensity

        gn_covG1_img -= gn_covG1_img.min()
        gn_covG1_img /= gn_covG1_img.max()

        return gn_covG1_img

    def convolutionG2(self, intensity):
        hor, verti = self.img.shape[0], self.img.shape[1]

        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        X, Y = torch.meshgrid(x2, x1)

        g2 = torch.cos(torch.sqrt(X ** 2 + Y ** 2))

        gauss_noise = torch.randn(129,129)
        covG2 = F.conv2d(gauss_noise.unsqueeze(0).unsqueeze(0), g2.unsqueeze(0).unsqueeze(0), padding=63)
        covG2 = covG2.squeeze()

        covG2 -= covG2.min()
        covG2 /= covG2.max()

        gn_covG2_img = self.img + covG2 * intensity
        gn_covG2_img -= gn_covG2_img.min()
        gn_covG2_img /= gn_covG2_img.max()

        return gn_covG2_img


    def convolutionG1V(self, intensity):
        '''
        The function generates correlated noise, with a convolution between a g funtion and a gaussian kernel.
        :param intensity:  The 'intensity' parameter refers to the intensity of the noise
        :return: it is returned the image with a convolutional noise
        '''
        img = self.img
        hor, verti = img.shape[0], img.shape[1]
        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        X1, X2 = torch.meshgrid(x1, x2)

        g1 = (1 - torch.abs(X2)).clamp(min=0) * (16 + torch.abs(X1)).clamp(min=0)

        g1 = g1.unsqueeze(0).unsqueeze(0)
        gauss_noise = torch.randn(129,129).unsqueeze(0).unsqueeze(0)

        covG1 = F.conv2d(gauss_noise, g1, padding=63)
        covG1 = covG1.squeeze(0).squeeze(0)

        covG1 -= covG1.min()
        covG1 /= covG1.max()

        gn_covG1v_img = img + covG1 * intensity

        gn_covG1v_img -= gn_covG1v_img.min()
        gn_covG1v_img /= gn_covG1v_img.max()

        return gn_covG1v_img



    def convolutionG1I(self, intensity, AMOUNT):
        hor, verti = self.img.shape[1], self.img.shape[0]

        # Precompute g1I
        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        X1, X2 = torch.meshgrid(x1, x2)
        g1I = (1 - torch.abs(X2)).clamp(min=0) * (16 + torch.abs(X1)).clamp(min=0)
        g1I = g1I.unsqueeze(0).unsqueeze(0)

        AMOUNT = AMOUNT
        SALT_VS_PEPPER = 0.6


        number_faulty_pixels = int(np.ceil(AMOUNT * self.img.numel()))
        indices = torch.randint(0, 129, (2, number_faulty_pixels))
        values = torch.tensor(np.random.binomial(1, SALT_VS_PEPPER, size=number_faulty_pixels), dtype=torch.float32)
        noise = torch.ones((129,129)) * 0.5
        noise[indices[0], indices[1]] = values * 1



        covG1I = F.conv2d(noise.unsqueeze(0).unsqueeze(0), g1I, padding=63)

        covG1I = covG1I.squeeze(0).squeeze(0)

        covG1I -= covG1I.min()
        covG1I /= covG1I.max()

        covG1I_img = covG1I * self.img
        covG1I_img -= covG1I_img.min()
        covG1I_img /= covG1I_img.max()

        im_covG1I_img = covG1I_img * intensity + self.img

        im_covG1I_img -= im_covG1I_img.min()
        im_covG1I_img /= im_covG1I_img.max()

        return im_covG1I_img

    def s1(self, intensityG1, intensityG2):

        hor, verti = self.img.shape[0], self.img.shape[1]
        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        [X, Y] = torch.meshgrid(x2, x1)

        g1 = (1 - torch.abs(X)).clamp(min=0) * (16 - torch.abs(Y)).clamp(min=0)
        g2 = torch.cos(torch.sqrt(X ** 2 + Y ** 2))

        g1 = g1.unsqueeze(0).unsqueeze(0)
        gauss_noise1 = torch.randn(129, 129).unsqueeze(0).unsqueeze(0)
        gauss_noise2 = torch.randn(129, 129).unsqueeze(0).unsqueeze(0)
        covG1 = F.conv2d(gauss_noise1, g1, padding=63)
        covG2 = F.conv2d(gauss_noise2, g2.unsqueeze(0).unsqueeze(0), padding=63)
        covG1 = covG1.squeeze(0).squeeze(0)
        covG2 = covG2.squeeze()
        covG2 -= covG2.min()
        covG2 /= covG2.max()
        covG1 -= covG1.min()
        covG1 /= covG1.max()
        gn_covG2_img = self.img + covG2 * intensityG2
        gn_covG1_img = self.img + covG1 * intensityG1
        s1_img = gn_covG1_img + gn_covG2_img
        s1_img -= s1_img.min()
        s1_img /= s1_img.max()
        return s1_img
    def s1Blur(self, intensityG1, intensityG2, sigma):

        hor, verti = self.img.shape[0], self.img.shape[1]
        x1 = torch.linspace(-verti // 2, verti // 2, verti)
        x2 = torch.linspace(-hor // 2, hor // 2, hor)
        [X, Y] = torch.meshgrid(x2, x1)

        g1 = (1 - torch.abs(X)).clamp(min=0) * (16 - torch.abs(Y)).clamp(min=0)
        g2 = torch.cos(torch.sqrt(X ** 2 + Y ** 2))

        g1 = g1.unsqueeze(0).unsqueeze(0)
        gauss_noise1 = torch.randn(129, 129).unsqueeze(0).unsqueeze(0)
        gauss_noise2 = torch.randn(129, 129).unsqueeze(0).unsqueeze(0)
        covG1 = F.conv2d(gauss_noise1, g1, padding=63)
        covG2 = F.conv2d(gauss_noise2, g2.unsqueeze(0).unsqueeze(0), padding=63)
        covG1 = covG1.squeeze(0).squeeze(0)
        covG2 = covG2.squeeze()
        covG2 -= covG2.min()
        covG2 /= covG2.max()
        covG1 -= covG1.min()
        covG1 /= covG1.max()
        gn_covG2_img = self.img + covG2 * intensityG2
        gn_covG1_img = self.img + covG1 * intensityG1
        s1_blur_img = gn_covG1_img + gn_covG2_img
        s1_blur_img = ski.filters.gaussian(
            s1_blur_img, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
        s1_blur_img = torch.from_numpy(s1_blur_img)

        s1_blur_img -= s1_blur_img.min()
        s1_blur_img /= s1_blur_img.max()

        return s1_blur_img

    def gaussianBlur(self, sigma):
        """
        The function applies the blur through a kernel, which in this case is a small matrix,
        and a convolution is performed with the image to obtain the blurry image
        :param sigma: he 'sigma' parameter refers to the standart deviation which determines the blur
        :return: it is returned the image with gaussian blur
        """
        gn_blur_img = ski.filters.gaussian(
            self.img, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
        gn_blur_img = torch.from_numpy(gn_blur_img)
        gn_blur_img -= gn_blur_img.min()
        gn_blur_img /= gn_blur_img.max()
        return gn_blur_img
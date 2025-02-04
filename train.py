import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np
from models.Attention_unet import AttU_Net
from degradationOperator import degradeBatch
import time
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
import math
from torch import multiprocessing
from models import PGGAN, utils, config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 128
Z_DIM = 256
IN_CHANNELS = 256
initial_learning_rate = 1e-2
decay_rate = 0.6

start = time.time()

model = AttU_Net(img_ch=1, output_ch=1).to(device)
gen = PGGAN.Generator(
    Z_DIM, IN_CHANNELS, img_channels=1
).to('cpu')

opt_gen = Adam(gen.parameters(), lr=1e-3, betas=(0.0, 0.99))
optimizer = Adam(model.parameters(),lr=initial_learning_rate)
lr_schedule  = ExponentialLR(optimizer, gamma=decay_rate,  last_epoch=-1)

utils.load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )

ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
loss_fn = torch.nn.L1Loss().to(device)
def log10(x):
  numerator = torch.log(x)
  denominator = torch.log(torch.tensor(10, dtype=numerator.dtype))
  return numerator / denominator
def PSNR(y_true, y_pred):
  msel = torch.mean((y_true - y_pred)**2)
  max_pixel=1
  return 10.0 * log10((max_pixel)/msel)
def loss_s(x, y):
    return (loss_fn(y,x)+ (1 - ssim(x,y)))
def degradedImages():

    x1 = []
    x2 = []
    mn = []
    sam = utils.generate_examples(gen, 5)
    sam = sam.squeeze()

    c = 0

    pbar = tqdm(sam)

    for i in pbar:
        c += 1
        start1 = time.time()
        dm, cln = degradeBatch(i)
        dm = dm.to(device)
        cln = cln.to(device)
        x1.append(dm)
        x2.append(cln)
        end1 = time.time()
        mn.append(end1 - start1)


    print("mean time image degradation: ", np.mean(mn))

    innd = torch.stack(x1, dim=0)
    innc = torch.stack(x2, dim=0)
    print("damages:", innd.shape)

    return innd, innc

def degradingImages(queue):
    x1 = []
    x2 = []
    mn = []
    sam = utils.generate_examples(gen, 5)
    sam = sam.squeeze()

    c = 0
    pbar = tqdm(sam)

    for i in pbar:
        c += 1
        start1 = time.time()
        dm, cln = degradeBatch(i)

        x1.append(dm)
        x2.append(cln)
        end1 = time.time()
        mn.append(end1 - start1)


    print("mean time image degradation: ", np.mean(mn))

    innd1 = torch.stack(x1, dim=0)
    innd2 = torch.stack(x2, dim=0)
    queue.put((innd1.numpy(), innd2.numpy()))
    print("new damages")

def train(queue,tot_epochs, epochs,  batch_size, i):

    if i > 1:
        model.load_state_dict(torch.load('guided_att_u.pt.pt'))
    model.train()

    tensor1_np, tensor2_np = queue.get()
    innd1 = torch.from_numpy(tensor1_np).to(device)
    innd2 = torch.from_numpy(tensor2_np).to(device)

    num_batches = math.ceil(innd1.size()[0] / batch_size)
    X_list = [innd1[batch_size * y:batch_size * (y + 1), :, :] for y in range(num_batches)]
    X_list2 = [innd2[batch_size * y:batch_size * (y + 1), :, :] for y in range(num_batches)]
    print("torch", X_list[0].shape)

    for epoch in range(epochs):
        print("\n epoch %d of %d" % (epoch + 1, epochs))
        print("Current learning rate:", optimizer.param_groups[0]['lr'])

        for d, c in tqdm(zip(X_list, X_list2), total=len(X_list)):

            cl = c.view(-1, 128, 128)

            dm = d.view(-1, 128, 128)
            y_batch_pred = model(dm.unsqueeze(1))
            clean = cl.unsqueeze(1)

            loss_value = loss_s(clean, y_batch_pred)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            cm = clean.to('cpu')
            pre = y_batch_pred.unsqueeze(1).to('cpu')
            psnrt = PSNR(cm, pre)
            '''
            if (epoch+1) % 10 == 0 or epoch == 0:
                with torch.no_grad():
                    model.eval()

                    rec = model(dm.unsqueeze(1))

                    dmg_im = dm.unsqueeze(1)

                    cln_im = cl.unsqueeze(1)

                    li = torch.concat((cln_im, rec, dmg_im), dim=0)

                save_image(li, f"epochs/image_epoch_{i}_{epoch}.jpg")
                model.train()
            '''
        if tot_epochs > 0 and tot_epochs % 200 == 0:
            lr_schedule.step()


     

        print(
            "(for 1 minibatch) Training loss %.7f | PSNR training %.7f"
            % (float(loss_value), float(psnrt))
        )
        torch.save(model.state_dict(), "guided_att_u.pt")


if __name__=='__main__':

    multiprocessing.set_start_method('spawn')

    innd1, innd2 = degradedImages()
    innd1_np = innd1.cpu().numpy()
    innd2_np = innd2.cpu().numpy()

    queue = multiprocessing.Queue()
    queue.put((innd1_np, innd2_np))

    epochs = 100
    batch_size = 5


    iterations = 4

    for i in range(iterations):
        print(i)
        tot_epochs=i*epochs
        pro1 = multiprocessing.Process(target=train, args=(queue,tot_epochs, epochs,batch_size,i))
        pro1.start()

        pro2 = multiprocessing.Process(target=degradingImages, args=(queue,))
        pro2.start()

        pro1.join()
        if i == (iterations-1):

            pro2.terminate()



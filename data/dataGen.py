import numpy as np
import matplotlib.pyplot as plt
import tqdm

#np.random.seed(0)

def gauss2D(xpts, ypts, mean=(0.0,0.0), var=None, normed=True):

    mx, my = mean
    if var is None:
        sx, sy = [np.min(xpts.shape)]*2
    else:
        sx, sy = var

    coeff = np.log(2*np.pi*np.sqrt(sx)*np.sqrt(sy))
    vx = (xpts - mx)**2/(2*sx)
    vy = (ypts - my)**2/(2*sy)

    logOfGauss = -coeff-vx-vy
    r = np.exp(logOfGauss)

    if normed:
        return r/r.max()
    else:
        return r

def addHFNoise(data, amp):
    noise = np.random.normal(size=data.size).reshape(data.shape)
    return data + amp*noise

def addLFNoise(data, amp, scale):
    x = np.linspace(-scale/2,scale/2,scale)
    y = np.linspace(-scale/2,scale/2,scale)
    X, Y = np.meshgrid(x,y)
    for i in range(10):
        loc = np.random.randint(-(scale/2),(scale/2),2)
        data += (amp/2.0)*gauss2D(X, Y, mean=loc, var=(scale*20, scale*20))

    return data

def addNoise(data, amp, scale):
    """Add noise to the data"""
    lfnData = addLFNoise(data, amp, scale)
    noisyData = addHFNoise(hfnData, amp)

    return noisyData

def create_dataset(mymean=np.array([7,-7]), myvar=np.array([20,20]), label_m=1, label_v=1, mysums=[]):

    print(mymean, myvar, 'oooooooooooooooooo')
    numelems = int(1e5)
    x = np.linspace(-14,14,28)
    y = np.linspace(-14,14,28)
    X, Y = np.meshgrid(x,y)
    dataset = np.ndarray((numelems,28*28))
    labels = np.ndarray((numelems))
    #varianceSmall = [np.array((5,5))]
    #varianceLarge = [np.array((20,20))]
    variances = [np.array((5,5)), np.array((20,20))]
    means = [np.array((-7,7)), np.array((7,-7))]
    #meansLeft = [np.array((-7,7))]
    #meansRight = [np.array((7,-7))]
    label_ids = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}

    var_index = np.random.choice(range(len(variances)))
    #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$', var_index)
    for i in tqdm.tqdm(range(numelems)):
        #var_index = np.random.choice(range(len(varianceSmall)))
        #mean_index = np.random.choice(range(len(meansLeft)))
        mean = mymean
        var = myvar + np.random.normal()
        labels[i] = label_ids[(label_m, label_v)]
        dataset[i] = gauss2D(X, Y, mean=mean, var=var).flatten()
        #print(mean, var)
    sum_data = np.sum(dataset)
    mysums.append([sum_data])
    print('******************\n', mysums)
    
    if label_m==1 and label_v==1:
        create_dataset(means[0], variances[0], 0, 0, mysums)
    elif label_m==0 and label_v==0:
        create_dataset(means[0], variances[1], 0, 1, mysums)
    elif label_m==0 and label_v==1:
        create_dataset(means[1], variances[0], 1, 0, mysums)
    else:
        pass

    np.save('single_gaussians_sizes=2_locs=2', dataset)
    np.save('single_gaussians_sizes=2_locs=2_labels', labels)


if __name__ == "__main__":
    create_dataset()

    #data = np.load('single_gaussians_sizes=2_locs=2.npy')
    #for i in range(10):
        #plt.imshow(data[i].reshape(28,28))
        #plt.show()
        #plt.clf()

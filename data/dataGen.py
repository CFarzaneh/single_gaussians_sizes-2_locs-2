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

def create_dataset():

    numelems = int(1e5)
    x = np.linspace(-14,14,28)
    y = np.linspace(-14,14,28)
    X, Y = np.meshgrid(x,y)
    dataset = np.ndarray((1000,28*28))

    groupOne = np.ndarray((numelems,28*28))
    groupTwo = np.ndarray((numelems,28*28))
    groupThree = np.ndarray((numelems,28*28))
    groupFour = np.ndarray((numelems,28*28))

    labels = np.ndarray((numelems))
    variances = [np.array((5,5)), np.array((20,20))]
    means = [np.array((-7,7)), np.array((7,-7))]
    label_ids = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}
    for i in tqdm.tqdm(range(1000)):
        var_index = np.random.choice(range(len(variances)))
        mean_index = np.random.choice(range(len(means)))
        mean = means[mean_index]
        var = variances[var_index] + np.random.normal()
        labels[i] = label_ids[(mean_index, var_index)]
        dataset[i] = gauss2D(X, Y, mean=mean, var=var).flatten()
    np.save('single_gaussians_sizes=2_locs=2', dataset)
    np.save('single_gaussians_sizes=2_locs=2_labels', labels)

    print("Generating groupOne (Right side, Multiple sizes)")
    for i in tqdm.tqdm(range(numelems)):
        if i%2 == 0:
            groupOne[i] = gauss2D(X, Y, mean=means[1], var=variances[0]+np.random.normal()).flatten()
        else:
            groupOne[i] = gauss2D(X, Y, mean=means[1], var=variances[1]+np.random.normal()).flatten()
    np.save('groupOne', groupOne)

    print("Generating groupTwo (Left side, Multiple sizes)")
    for i in tqdm.tqdm(range(numelems)):
        if i%2 == 0:
            groupTwo[i] = gauss2D(X, Y, mean=means[0], var=variances[0]+np.random.normal()).flatten()
        else:
            groupTwo[i] = gauss2D(X, Y, mean=means[0], var=variances[1]+np.random.normal()).flatten()
    np.save('groupTwo', groupTwo)

    print("Generating groupThree (Multiple sides, Large size)")
    for i in tqdm.tqdm(range(numelems)):
        if i%2 == 0:
            groupThree[i] = gauss2D(X, Y, mean=means[0], var=variances[1]+np.random.normal()).flatten()
        else:
            groupThree[i] = gauss2D(X, Y, mean=means[1], var=variances[1]+np.random.normal()).flatten()
    np.save('groupThree', groupThree)

    print("Generating groupFour (Multiple sides, Small size)")
    for i in tqdm.tqdm(range(numelems)):
        if i%2 == 0:
            groupFour[i] = gauss2D(X, Y, mean=means[0], var=variances[0]+np.random.normal()).flatten()
        else:
            groupFour[i] = gauss2D(X, Y, mean=means[1], var=variances[0]+np.random.normal()).flatten()
    np.save('groupFour', groupFour)

    # data = np.load('groupFour.npy')
    # for i in range(10):
    #     plt.imshow(data[i].reshape(28,28))
    #     plt.show()
    #     plt.clf()


if __name__ == "__main__":

    create_dataset()

    # data = np.load('single_gaussians_sizes=2_locs=2.npy')
    # for i in range(10):
    #     plt.imshow(data[i].reshape(28,28))
    #     plt.show()
    #     plt.clf()

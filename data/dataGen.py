import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

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


def create_dataset(mymean, myvar, mysums, counter):

    numelems = int(1e5)
    x = np.linspace(-14,14,28)
    y = np.linspace(-14,14,28)
    X, Y = np.meshgrid(x,y)
    dataset = np.ndarray((5,numelems,28*28))
    var_inc = 0.4                          #Inc var by 0.4 
    #labelss = np.ndarray((numelems))
    #label_ids = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}

    #var_index = np.random.choice(range(len(myvar)))
    
    for i in range(5):      #ups variance
        for j in range(numelems):
            if counter < 3:
                mean = mymean + np.random.normal(scale=var_inc*i) # vary loc from 0.0 to 1.6, leave size alone
                var = myvar
            else:
                mean = mymean
                var = myvar + np.random.normal(scale=var_inc*i)   # vary size the same way, leave loc
            #labelss[i] = label_ids[labels[0]]
            dataset[i,j] = gauss2D(X, Y, mean=mean, var=var).flatten()

    sum_data = np.sum(dataset, axis=1)
    mysums.append([sum_data])

    return mysums, dataset


if __name__ == "__main__":
    
    variances = [np.array((5,5)), np.array((20,20))]
    means = [np.array((-7,7)), np.array((7,-7))]
    mysums = []
    #labels = [[] for _ in range(100000)]
    counter = 1

    RLbig = []
    RLsmall = []
    rightBS = []
    leftBS = []

    for i in range(2):
        for j in range(2):
            print(counter)
            #labels[i] = [(i, j)]
            mysums, data = create_dataset(means[i],variances[j], mysums, counter)
            #np.save('single_gaussians_sizes=2_locs=2_%d'%(counter), data)
            #np.save('single_gaussians_sizes=2_locs=2_labels_%d'%(counter), labelss)

            if counter is 1:
                RLsmall.append(data)
                leftBS.append(data) 
            if counter is 2:
                RLbig.append(data) 
                leftBS.append(data) 
            if counter is 3:
                RLsmall.append(data) 
                rightBS.append(data) 
            if counter is 4:
                RLbig.append(data) 
                rightBS.append(data)

                np.save('single_gaussians_RLbig', RLbig)
                rlb_labels = np.ndarray((5,100000))
                rlb_labels.fill(1)
                np.save('single_gaussians_RLbig', rlb_labels)

                np.save('single_gaussians_RLsmall', RLsmall)
                rls_labels = np.ndarray((5,100000))
                rls_labels.fill(2)
                np.save('single_gaussians_RLsmall', rls_labels)

                np.save('single_gaussians_rightBS', rightBS)
                rbs_labels = np.ndarray((5,100000))
                rbs_labels.fill(3) 
                np.save('single_gaussians_RLbig', rlb_labels)

                np.save('single_gaussians_leftBS', leftBS)
                lbs_labels = np.ndarray((5,100000))
                lbs_labels.fill(4)
                np.save('single_gaussians_RLbig', lbs_labels)

            counter+=1

    #print(mysums, len(mysums))

    #data = np.load('single_gaussians_sizes=2_locs=2.npy')
    #for i in range(10):
        #plt.imshow(data[i].reshape(28,28))
        #plt.show()
        #plt.clf()

import numpy as np
import matplotlib.pyplot as plt
import tqdm

def calculate_means():
	numelems = int(1e5)
	
	data = np.load('single_gaussians_sizes=2_locs=2.npy')
	tot_data = np.reshape(data, (numelems,28,28))
	tot_mean = tot_data.mean(0)
	
	plt.imshow(tot_mean)
	plt.show()

	groupOne = np.load('groupOne.npy')
	groupOne_data = np.reshape(groupOne, (numelems,28,28))
	groupOne_mean = groupOne_data.mean(0)

	plt.imshow(groupOne_mean)
	plt.show()

	groupTwo = np.load('groupTwo.npy')
	groupTwo_data = np.reshape(groupTwo, (numelems,28,28))
	groupTwo_mean = groupTwo_data.mean(0)

	plt.imshow(groupTwo_mean)
	plt.show()

	groupThree = np.load('groupThree.npy')
	groupThree_data = np.reshape(groupThree, (numelems,28,28))
	groupThree_mean = groupThree_data.mean(0)

	plt.imshow(groupThree_mean)
	plt.show()

	groupFour = np.load('groupFour.npy')
	groupFour_data = np.reshape(groupFour, (numelems,28,28))
	groupFour_mean = groupFour_data.mean(0)
	
	plt.imshow(groupFour_mean)
	plt.show()

if __name__ == "__main__":
	calculate_means()

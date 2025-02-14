import os
import torch
from torchvision import datasets
from torchvision import transforms 
import snntorch as snn
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def data_preprocessing(dataset, dir, split_params, kernel_params):

  # The function should check if the data directory exists in the file tree
  # If it doesn't, create the folder "data", with subfolders "processed", "raw", and "split"
  # Run datasets.FashionMNIST with the dir + "raw", the datasets method will avoid downloading 
  # the file if it has already been downloaded, so you don't have to do anything 
  # Look for a split_params file within dir + "split"
  # If the split_params file exists, read the split present 

  print("Running data_preprocessing function")
  if dataset != "FashionMNIST":
    print("ERROR: Currently only compatible with FashionMNIST")
    return None
  
  if os.path.exists(dir):
    print("Directory already exists")
  else:
    print("Creating directory")
    os.mkdir(dir)
    os.mkdir(dir + '/' + 'raw')
    os.mkdir(dir + '/' + 'split')
    open(dir + '/' + 'split' + '/' + 'split_params.txt', 'x')
    os.mkdir(dir + '/' + 'processed')
    open(dir + '/' + 'processed' + '/' + 'processed_params.txt', 'x')

  raw_train = datasets.FashionMNIST(root=dir + '/' + 'raw', train=True, download=True, transform=None)
  raw_test = datasets.FashionMNIST(root=dir + '/' + 'raw', train=False, download=True, transform=None)

  n_train, n_test, r_seed = split_params 

  with open(dir + '/' + 'split' + '/' + 'split_params.txt', 'w') as s:
    s.write(f"{n_train}\n")
    s.write(f"{n_test}\n")
    s.write(f"{r_seed}\n")

  split_train, split_test = split_sets(raw_train, raw_test, split_params)
  torch.save(split_train, dir + '/' + 'split' + '/' + 'train.pt')
  torch.save(split_test, dir + '/' + 'split' + '/' + 'test.pt')

  dim, ang, ppa, ctr, sur = kernel_params

  ON_kernel = DOG_kernel(dim, ang, ppa, ctr, sur)
  OFF_kernel = DOG_kernel(dim, ang, ppa, ctr, sur)

  ON_kernel.setFilterCoefficients(ONOFF="ON")
  OFF_kernel.setFilterCoefficients(ONOFF="OFF")

  
  return None




class DOG_kernel:
  def __init__(self, dim, ang, ppa, ctr, sur):
    """
    Creates a dogKernel object representing a kernel with passed overall, centre, and surround dimensions.

    :param dim: Kernel dimensions in pixels (size of the kernel object).
    :param ang: Kernel dimensions in angular units (used for scaling or angular representation).
    :param ctr: Size of the center within the kernel (in pixels).
    :param sur: Size of the surround within the kernel (in pixels).
    :object kernel: The actual kernel itself, an np.array set with setFilterCoefficients
    """
    self.dim = dim
    self.ang = ang
    self.ppa = ppa
    self.ctr = ctr
    self.sur = sur
    self.kernel = None

  def __str__(self):
    return f"___________________________\nKernel:\n{self.kernel}\nKernel Size: {np.shape(self.kernel)}\nKernel Center: {self.ctr}\nSurround Center: {self.sur}"

  def setFilterCoefficients(self, ONOFF):
    """
    Sets the filter coefficents of the dogKernel using the dogKernel's attributes (dim, ang, ctr, sur)
    :param self:
    :param ONOFF: ON sets the kernel with a center-on/surround-off distribution, OFF sets the kernel with a center-off/surround-on distribution
    :return self.kernel: Sets the filter coefficients of the dogKernel with a difference of gaussian distribution
    """

    ctr_kernel = self.gen_gaussian_kernel(self.dim, self.ctr)
    sur_kernel = self.gen_gaussian_kernel(self.dim, self.sur)

    if ONOFF == 'ON':
      self.kernel = +ctr_kernel - sur_kernel
    elif ONOFF == 'OFF':
      self.kernel = -ctr_kernel + sur_kernel
    else:
      print(f"Incorrect argument given, passed {ONOFF} to ONOFF, when it should be 'ON' or 'OFF'")

    self.kernel = self.kernel - np.mean(self.kernel)

    input_min = 0
    input_max = 1

    self.kernel = self.kernel / (np.sum(np.abs(self.kernel)) / 2) * (input_max - input_min)

  def gen_gaussian_kernel(self, shape, sigma):
    """
    Creates a 2D Gaussian kernel.
    :param shape: size of the kernel (np.array)
    :param sigma: standard deviation for the Gaussian
    :return: Gaussian kernel (np.array)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h /= h.sum()  # Normalize to sum to 1
    return h
  
  def displayKernel(self, show_plt=True, show_hist=True, show_img=True):
    """
    Kernel display function
    :param plt: toggle pyplot (bool)
    :param hist: toggle histogram (bool)
    :param img: toggle example img, Fashion-MNIST boot (bool)
    """
    display = plt.figure()

    if show_plt == True:
      display.add_subplot(1,3,1)
      plt.imshow(self.kernel, cmap='grey')
      plt.title("Kernel")

    if show_hist == True:
      display.add_subplot(1,3,2)
      plt.hist(self.kernel.flatten(), bins=20, range=[np.min(self.kernel), np.max(self.kernel)])
      plt.title("Histogram")
    
    if show_img == True:
      display.add_subplot(1,3,3)
      test_img = np.load('test_img.npy')
      test_img = np.clip(convolve2d(test_img, self.kernel, mode='same'), 0, None)
      plt.imshow(test_img, cmap='grey')
      plt.title("Image")


    plt.show()


def split_sets(raw_train, raw_test, split_params):

  n_train, n_test, r_seed = split_params
  r_seed = np.random.default_rng(r_seed)

  # training_set = utils.data_subset(training_images, num_train_samples)
  # testing_set = utils.data_subset(testing_images, num_test_samples)
  
  train_idx = r_seed.choice(len(raw_train), n_train, replace=False)
  test_idx = r_seed.choice(len(raw_test), n_test, replace=False)

  split_train = []
  split_test = [] 

  for idx in train_idx:
    split_train.append(raw_train[idx])
  
  for idx in test_idx:
    split_test.append(raw_test[idx])

  return split_train, split_test

def dataAnalysis(dataset):

  labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
  px_val = []
  num_label = [0] * len(labels)

  for curr in dataset:
    img, label = curr
    img = np.array(img)
    px_val.extend(img.flatten())
    num_label[label] += 1

  px_val = np.array(px_val)

  results_dict = {
    "min" : np.min(px_val),
    "max" : np.max(px_val),
    "mean" : np.mean(px_val),
    "median" : np.median(px_val),
    "std_val" : np.std(px_val),
    "px_val" : px_val,
    "label_plot" : [labels, num_label]
  }

  return results_dict

def genLGNActivityMaps(data, DoGkernel, debug=False):
  kernel = DoGkernel
  convolved_dataset = []


  for curr in data:
    img, label = curr
    img = np.array(img) # Convert to np.array
    img = convolve2d(img, kernel, mode="same") # Apply DoG Kernel
    img = np.clip(img, 0, None) # Clip negative values 
    new_tuple = (img, label) # Create tuple 
    convolved_dataset.append(new_tuple) # Append tuple to list 

  if debug == True:
    # Debug statements

    print(f"Length of new dataset: {len(convolved_dataset)}")
    print(f"Type of new dataset: {type(convolved_dataset)}")
    print(f"Type of new dataset element: {type(convolved_dataset[0])}")
    print(f"Type of new dataset element first element: {type(convolved_dataset[0][0])}")
    print(f"Type of new dataset element second element: {type(convolved_dataset[0][1])}")
    images = [curr[0] for curr in convolved_dataset]
    labels = [curr[1] for curr in convolved_dataset]

    print(f"Min of new dataset element first element: {np.min(images)}")
    print(f"Max of new dataset element first element: {np.max(images)}")
    print(f"Min of new dataset element second element: {np.min(labels)}")
    print(f"Max of new dataset element second element: {np.max(labels)}")

  return convolved_dataset
    
def convertToTensor(data):
  tensor_dataset = []
  
  for curr in data:
    img, label = curr
    img = torch.from_numpy(img).float() / 255  # Convert numpy array to tensor and ensure it's float
    
    new_tuple = (img, label)
    tensor_dataset.append(new_tuple)

  return tensor_dataset


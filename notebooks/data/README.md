# Neural Recordings for Training the Mouse Model

### Data Files:
- `FX8_nat60k_2023_05_16.npz`
- `FX9_nat60k_2023_05_15.npz`
- `FX10_nat60k_2023_05_16.npz`
- `FX20_nat60k_2023_09_29.npz`
- `L1_A1_nat60k_2023_03_06.npz`
- `L1_A5_nat60k_2023_02_27.npz`

Each file contains the recordings of training and testing images from one mouse, and includes the following keys:
- `["sp"]`: Shape (n_neurons, n_images), responses of each neuron to each image used for training and validating.
- `["istim_sp"]`: Shape (n_images), indexes of the images in the image file, corresponding to the stimuli used in "sp".
- `["ss_all"]`: Shape (n_test_images, n_repeats, n_neurons), responses of each neuron to each test image (500 images).
- `["istim_ss"]`: Shape (n_test_images), indexes of the test images in the image file.
- `["xpos"]`: Shape (n_neurons), x cortical position of each neuron.
- `["ypos"]`: Shape (n_neurons), y cortical position of each neuron.
- `["iplane"]`: Shape (n_neurons), imaging plane the neuron is from.

# Image File

- `nat60k_text16.mat`: Contains images with shape (66, 264, 68000), including 30,000 nature images and their flipped versions used for model training, and 8000 texture images for invariance analysis. Each image has a shape of (66, 264).

# Neural Recordings of Texture Dataset Used for Invariance Analysis

### Data Files:
- `text16_FX8_2023_05_16.npz`
- `text16_FX9_2023_05_15.npz`
- `text16_FX10_2023_05_16.npz`
- `text16_FX20_2023_09_29.npz`

Each file contains the recordings of texture images from 16 categories from one mouse, and includes the following keys:
- `["sp"]`: Shape (n_neurons, n_images), responses of each neuron to each image. Includes 4800 images, with 300 images from each of the 16 texture categories used for training the classifier. Contains the same number of neurons as the neural recordings for model training.
- `["istim_sp"]`: Shape (n_images), indexes of the images in the image file, corresponding to the stimuli used in "sp".
- `["labels"]`: Shape (n_images), label of the category each image belongs to.
- `["ss_all"]`: Shape (n_test_images, n_repeats, n_neurons), responses of each neuron to each test image (50 images per category).
- `["istim_ss"]`: Shape (n_test_images), label of the category each test image belongs to.
- `["ss_labels"]`: Shape (n_test_images), indexes of the test images in the image file.
- `["xpos"]`: Shape (n_neurons), x cortical position of each neuron.
- `["ypos"]`: Shape (n_neurons), y cortical position of each neuron.

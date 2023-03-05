from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from skimage import feature
from utils import mnist_reader

X, Y = mnist_reader.load_mnist("data/fashion", kind="train")
types = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# Select 20 sample images
sample_imgs = X[:20]

# Reshape the sample images to 28x28 arrays
sample_imgs = [img.reshape(28, 28) for img in sample_imgs]

# Compute the HOG transform of the images
fd_images = []
for i, img in enumerate(sample_imgs):
    fd, hog_image = feature.hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        visualize=True,
    )
    fd_images.append(hog_image)

# Plot the original images and HOG transforms
for i in range(20):
    plt.subplots_adjust(hspace=0.5)
    # plt.subplot(4, 5, i + 1)
    # plt.imshow(sample_imgs[i], cmap="gray")
    # plt.title(f"Type: {types[Y[i]]}", fontsize=8)
    plt.subplot(4, 5, i + 1)
    plt.imshow(fd_images[i], cmap="gray")
    plt.title(f"HOG[{types[Y[i]]}]", fontsize=8)
plt.show()

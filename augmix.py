import torch
import numpy as np
import augmentations

MIXTURE_WIDTH = 3

def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    # aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * MIXTURE_WIDTH))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(MIXTURE_WIDTH):
        image_aug = image.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, -1)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

class AugMixDataset(torch.utils.data.Dataset):
	"""Dataset wrapper to perform AugMix augmentation."""
	def __init__(self, dataset, preprocess, no_jsd=False):
		self.dataset = dataset
		self.preprocess = preprocess
		self.no_jsd = no_jsd

	def __getitem__(self, i):
		x, y = self.dataset[i]
		if self.no_jsd:
			return aug(x, self.preprocess), y
		else:
			im_tuple = (self.preprocess(x), aug(x, self.preprocess), aug(x, self.preprocess))
			return im_tuple, y

	def __len__(self):
		return len(self.dataset)
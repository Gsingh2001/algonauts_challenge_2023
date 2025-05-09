Thought for a few seconds


```markdown
# Reconstructing Natural Scene Images from fMRI using a Conditional DCGAN

This repository contains code and documentation for our conditional DCGAN model that reconstructs low-resolution natural scene images from functional MRI (fMRI) signals. We present two experiments:

1. **Experiment 1**: Unconditional DCGAN trained on images only, with a simple fMRI-to-latent mapping for preliminary reconstructions.  
2. **Experiment 2**: Full end-to-end conditional DCGAN trained on paired image–fMRI data.

The code, Jupyter notebooks, and preprocessed data are organized for easy reproduction of our results.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Features](#features)  
- [Installation](#installation)  
- [Data Preparation](#data-preparation)  
- [Usage](#usage)  
  - [Experiment 1](#experiment-1)  
  - [Experiment 2](#experiment-2)  
- [Results](#results)  
- [Dependencies](#dependencies)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)  

---

## Project Structure

```

.
├── data/
│   ├── images/             # Downsampled 32×32 stimulus images
│   └── fmri\_pca/           # PCA-reduced fMRI features (1024D .npy files)
│
├── experiments/
│   ├── exp1\_baseline.ipynb # Notebook for Experiment 1 (unconditional GAN)
│   └── exp2\_conditional.ipynb # Notebook for Experiment 2 (conditional DCGAN)
│
├── models/
│   ├── generator.py        # DCGAN generator class
│   ├── discriminator.py    # DCGAN discriminator class
│   └── train.py            # Training scripts
│
├── results/
│   ├── figures/            # Generated sample images & loss curves
│   └── logs/               # Training logs & checkpoints
│
├── README.md               # This file
└── requirements.txt        # Python dependencies

````

---

## Features

- **Data preprocessing**: PCA reduction of fMRI vectors to 1024 components; image resizing to 32×32.  
- **Experiment 1**: Unconditional DCGAN trained on images, plus a simple fMRI→latent mapping for coarse reconstructions.  
- **Experiment 2**: Conditional DCGAN that concatenates PCA features with noise in the generator and discriminator.  
- **Visualization**: Loss curves, real vs. generated image montages, and real vs. reconstructed comparisons.  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/fmri-dcgan-reconstruction.git
   cd fmri-dcgan-reconstruction
````

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. **Download the Algonauts 2023 (NSD) dataset** following the instructions at [Algonauts Challenge](https://algonauts.github.io/).
2. **Extract** the stimulus images and fMRI responses for Subject 1.
3. **Resize images** to 32×32 RGB and save in `data/images/`.
4. **Standardize** and **PCA-reduce** the fMRI vectors to 1024 components:

   ```python
   from sklearn.decomposition import PCA
   import numpy as np

   fmri = np.load('raw_fmri.npy')       # shape (N_samples, N_voxels)
   pca = PCA(n_components=1024).fit(fmri)
   fmri_pca = pca.transform(fmri)
   np.save('data/fmri_pca/features.npy', fmri_pca)
   ```

---

## Usage

### Experiment 1

1. **Open** `experiments/exp1_baseline.ipynb` in Jupyter.
2. **Run** all cells to:

   * Train the unconditional DCGAN for 50 epochs.
   * Visualize loss curves and generated samples.
   * Perform a simple fMRI→latent mapping and display preliminary reconstructions.

### Experiment 2

1. **Open** `experiments/exp2_conditional.ipynb` in Jupyter.
2. **Run** all cells to:

   * Train the conditional DCGAN for 500 epochs on image–fMRI pairs.
   * Plot training losses (generator vs.\ discriminator).
   * Generate final samples, real-vs-fake comparisons, and real-vs-reconstructed montages.

---

## Results

Figures and checkpoints are saved under `results/`:

* **Loss curves**: `results/figures/exp1_loss.png`, `exp2_loss.png`
* **Generated samples**: `results/figures/exp1_samples.png`, `exp2_samples.png`
* **Reconstruction comparisons**: `results/figures/exp1_recon.png`, `exp2_recon.png`

Checkpoints for generator and discriminator weights are under `results/checkpoints/`.

---

## Dependencies

All major dependencies are listed in `requirements.txt`. Key packages include:

* `torch`
* `torchvision`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `tqdm`

Install via:

```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! Please:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add awesome feature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

* Allen, E.\~J., Petersen, K., & Carandini, M. (2022). *A massive 7T fMRI dataset to bridge cognitive neuroscience and AI.* Nature Neuroscience, 25(1), 116–126.
* Güçlütürk, Y., Güçlü, U., Seeliger, K., et al. (2017). *Reconstructing perceived faces from brain activations with deep adversarial neural decoding.* NeurIPS.
* Miyawaki, Y., Uchida, H., Yamashita, O., et al. (2008). *Visual image reconstruction from human brain activity using multiscale local image decoders.* Neuron, 60(5), 915–929.
* Nishimoto, S., Vu, A.\~T., Naselaris, T., et al. (2011). *Reconstructing visual experiences from brain activity evoked by natural movies.* Current Biology, 21(19), 1641–1646.
* Ozcelik, F., & van Rullen, R. (2023). *Natural scene reconstruction from fMRI signals using latent diffusion.* Scientific Reports, 13, 15666.
* Radford, A., Metz, L., & Chintala, S. (2016). *Unsupervised representation learning with deep convolutional generative adversarial networks.* ICLR.
* Seeliger, K., Güçlü, U., Ambrogioni, L., et al. (2018). *Generative adversarial networks for reconstructing natural images from brain activity.* NeuroImage, 181, 775–785.
* Shen, G., Dwivedi, K., Majima, K., et al. (2019). *End-to-end deep image reconstruction from human brain activity.* Frontiers in Computational Neuroscience, 13, 21.

---

```

```

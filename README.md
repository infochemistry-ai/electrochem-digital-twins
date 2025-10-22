# Electrochemical Digital Twin
This repository implements digital twin models that generate synthetic cyclic voltammetry (CV) profiles for corrosion inhibitor systems. A digital twin is a virtual copy of a physical process that can be used to augment experiments and accelerate development. In this project, experimental CV curves are combined with ab initio density‑functional theory (DFT) descriptors and cheminformatics descriptors from RDKit to train generative deep‑learning models. The models produce realistic voltammograms that emulate electrochemical measurements and support research into corrosion inhibitors.

## Installation steps

**1. Clone the repository:**
  ```bash
  git clone https://github.com/infochemistry-ai/electrochem-digital-twins.git
  cd electrochem-digital-twins
  ```
  
**2. Create a virtual environment and install dependencies:**
  ```bash
  uv sync  # resolves and installs dependencies defined in pyproject.toml
  source .venv/bin/activate
  ```
**3.(Optional) Configure S3 credentials.** Create a .env file in the project root with the following keys so that the download_data.py script can authenticate to Yandex S3:
  ```bash
  S3_ACCESS_KEY=your_access_key
  S3_SECRET_KEY=your_secret_key
  ```

## Data
The experimental CV curves and descriptors are stored in a Yandex S3 bucket. [Download data](https://storage.yandexcloud.net/digital-twin-data/data_dd.zip)

## Preprocessing

Data preprocessing is encapsulated in the src/data/preprocessing/pipeline.py class. It performs:

 - **Tail removal and interpolation** – missing tails in the voltage/current curves are filled and then wavelet interpolation standardises each voltammogram to 968 samples.

 - **Descriptor merging and normalisation** – descriptors from RDKit (mol_rdkit.csv) and DFT calculations (mol_dft.csv) are merged on the inhibitor name; optional min–max scaling is available in the pipeline.

 - **Feature extraction** – the extractor.py module calculates electrochemical metrics such as anodic/cathodic peak positions, areas and the half‑wave potential.

 - **Train/test splitting** – splitter.py supports leaving one inhibitor out for validation; for example, training on all inhibitors except benzotriazole and testing on it.


## Training models

Three generative architectures are available:

  1. **Convolutional VAE (conv‑VAE)** – defined in src/models/vaeconv.py. A 1‑D convolutional encoder compresses the voltammogram into a latent vector and a decoder reconstructs it given additional descriptor information.

  2. **LSTM‑based VAE (LSTM‑VAE)** – defined in src/models/vaelstm.py. This model uses recurrent encoders/decoders and teacher forcing for sequence generation. In the associated paper, the LSTM‑VAE achieved the best similarity to experimental data (lowest DTW distance).

  3. **Convolutional GAN (conv‑GAN)** – the generator and discriminator are defined in src/train/trainer/TrainerGAN.py and related modules. The GAN sometimes suffers from training instabilities and higher error compared with the VAE approaches.

Each model has an entry‑point script in src/train/start_training/:
  ```bash
  python src/train/start_training/train_vae_conv.py    # trains the convolutional VAE
  python src/train/start_training/train_vae_lstm.py    # trains the LSTM‑VAE
  python src/train/start_training/train_gan_conv.py    # trains the convolutional GAN
  ```

By default these scripts loop over a list of inhibitors, leave one inhibitor out for validation, create DataLoader objects from CVADataset, and train for 500 epochs. Training and validation losses (including reconstruction loss and KL divergence for VAEs) are logged. The scripts save model weights, loss tables and plots in the reports/ directory.


## Inference

After training, you can generate new CV curves using the trained models. Example notebooks are provided in notebooks/inference_example/. These notebooks load a trained model checkpoint, sample from the latent space (for the GAN, random noise combined with descriptor features; for the VAEs, sampling from the latent distribution) and plot the generated voltammograms. Dynamic Time Warping (DTW) is recommended for quantitative comparison with experimental curves.

## Contributing

Contributions are welcome! If you would like to add new models, improve the preprocessing pipeline or extend the documentation:

1. Fork the repository and create your branch (git checkout -b feature/my-feature).

2. Commit your changes and ensure that they follow the existing code style.

3. Submit a pull request describing the changes and referencing any relevant issues.

Please note that this project is released under the MIT License, so by contributing you agree that your contributions will be licensed under the same terms.


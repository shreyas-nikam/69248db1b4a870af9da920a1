
# Technical Specification for Jupyter Notebook: SynthFin Data Generator: VAEs & GANs

## 1. Notebook Overview

**Learning Goals:**
This notebook aims to equip Financial Data Engineers with the practical skills and theoretical understanding required to implement and compare Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) for generating synthetic financial data. Upon completing this lab, users will be able to:
*   Understand the theoretical foundations of VAEs and GANs, including their core components and operational principles.
*   Implement a VAE model, defining its encoder, decoder, and a comprehensive loss function that balances reconstruction error and regularization.
*   Implement a GAN model, comprising a Generator and a Discriminator, and manage its game-theoretic training loop.
*   Execute and observe a simplified training process for both VAE and GAN architectures, tracking key performance metrics.
*   Generate new synthetic financial data using trained VAE and GAN models.
*   Perform comparative analysis of statistical properties (mean, variance, distribution shape) and time series characteristics between real and synthetic data.
*   Explore the impact of adjusting latent variables on synthetic data generation, particularly for VAEs.
*   Grasp the utility of VAEs and GANs in financial applications such as privacy-preserving data sharing, stress-testing financial models, and market scenario simulation.

**Target Audience:**
The notebook is specifically targeted at **Financial Data Engineers**. This persona is expected to have a foundational understanding of machine learning concepts, neural networks, and financial data characteristics. The content will balance theoretical explanations with practical implementation details, focusing on relevance to financial applications.

## 2. Code Requirements

### List of Expected Libraries

*   `numpy`: For numerical operations and data generation.
*   `pandas`: For data manipulation and structuring.
*   `torch`: The primary deep learning framework for building VAE and GAN models.
*   `torch.nn`: For defining neural network layers.
*   `torch.optim`: For optimization algorithms (e.g., Adam).
*   `torch.utils.data`: For creating DataLoaders.
*   `matplotlib.pyplot`: For generating static plots and visualizations.
*   `seaborn`: For enhanced statistical data visualization.
*   `scipy.stats`: For statistical tests, such as the Kolmogorov-Smirnov test.
*   `sklearn.preprocessing`: For data scaling and normalization (e.g., `MinMaxScaler`, `StandardScaler`).
*   `sklearn.manifold`: Potentially for dimensionality reduction for latent space visualization (e.g., `TSNE`).
*   `sklearn.decomposition`: Potentially for dimensionality reduction for latent space visualization (e.g., `PCA`).

### List of Algorithms or Functions to be Implemented (without code)

*   `generate_mock_financial_time_series_data`: Function to create a synthetic financial dataset simulating stock prices or interest rates.
*   `preprocess_financial_data`: Function to apply scaling (e.g., Min-Max scaling) to the input financial time series data and reshape for sequence models.
*   `build_vae_encoder`: Function to define the architecture of the VAE encoder using `torch.nn.Module`, typically consisting of linear layers or recurrent layers (e.g., GRU/LSTM) followed by linear layers to output mean and log-variance for the latent space.
*   `build_vae_decoder`: Function to define the architecture of the VAE decoder using `torch.nn.Module`, typically consisting of linear layers or recurrent layers to reconstruct the input data from a latent space sample.
*   `reparameterize`: Function implementing the reparameterization trick for VAEs, sampling from $N(\mu, \sigma^2)$ using $z = \mu + \sigma \cdot \epsilon$.
*   `vae_loss_function`: Function to calculate the VAE loss, which is a combination of reconstruction loss (e.g., Mean Squared Error) and KL divergence loss.
*   `train_vae_epoch`: Function to perform one epoch of VAE training, including forward pass, loss calculation, backward pass, and optimizer step.
*   `train_vae_model`: Orchestrates the VAE training over multiple epochs, calls `train_vae_epoch`, and collects loss history.
*   `generate_synthetic_data_vae`: Function to generate synthetic data by sampling from the VAE's latent space and passing it through the decoder.
*   `build_gan_generator`: Function to define the architecture of the GAN generator using `torch.nn.Module`, taking random noise as input and producing synthetic data.
*   `build_gan_discriminator`: Function to define the architecture of the GAN discriminator using `torch.nn.Module`, taking data (real or synthetic) as input and outputting a probability of it being real.
*   `gan_generator_loss`: Function to calculate the loss for the GAN generator (e.g., binary cross-entropy aiming for the discriminator to classify fake data as real).
*   `gan_discriminator_loss`: Function to calculate the loss for the GAN discriminator (e.g., binary cross-entropy distinguishing real from fake data).
*   `train_gan_epoch`: Function to perform one epoch of GAN training, including discriminator and generator steps.
*   `train_gan_model`: Orchestrates the GAN training over multiple epochs, calls `train_gan_epoch`, and collects loss history.
*   `generate_synthetic_data_gan`: Function to generate synthetic data by sampling random noise and passing it through the trained GAN generator.
*   `plot_loss_curves`: Function to plot the training loss curves for VAE (reconstruction, KL divergence) and GAN (generator, discriminator).
*   `plot_feature_distributions`: Function to visualize the distribution of selected features (e.g., using histograms or KDE plots) for real and synthetic data.
*   `plot_synthetic_and_real_time_series`: Function to plot a selection of real and synthetic time series side-by-side for visual comparison.
*   `calculate_statistical_metrics`: Function to compute and display descriptive statistics (mean, standard deviation) for real and synthetic data features.
*   `perform_kolmogorov_smirnov_test`: Function to perform the Kolmogorov-Smirnov (K-S) test to quantitatively compare the distributions of real and synthetic data features.
*   `generate_with_adjusted_latent`: Function to generate synthetic data by perturbing specific dimensions of the VAE's latent space and passing it through the decoder.
*   `visualize_latent_space`: (Optional) Function to apply dimensionality reduction (e.g., t-SNE or PCA) to the VAE's latent space and plot the results.

### Visualization Requirements

1.  **VAE Training Loss Curves:** A line plot showing VAE reconstruction loss and KL divergence loss over training epochs.
2.  **GAN Training Loss Curves:** A line plot showing GAN generator loss and discriminator loss over training epochs.
3.  **Data Distribution Comparison (Histograms/KDEs):** Multiple plots (e.g., a grid of subplots) displaying histograms or Kernel Density Estimates (KDEs) for 2-3 key features of the original real dataset, VAE-generated synthetic data, and GAN-generated synthetic data. These should be overlaid or presented side-by-side for easy comparison.
4.  **Time Series Plots:** Multiple line plots, each comparing a sample real time series with a corresponding VAE-generated and GAN-generated synthetic time series. These plots should illustrate the temporal dynamics and realism of the generated data.
5.  **Latent Space Exploration (Optional for VAE):** A 2D scatter plot (if applicable, using t-SNE or PCA for reduction) of the VAE's latent space, potentially colored by some inferred characteristic or simply showing the distribution of latent vectors.
6.  **Adjusted Latent Variable Plots (for VAE):** Line plots showing multiple synthetic time series generated by systematically varying one or two dimensions of the VAE's latent space, demonstrating the model's generative control.

## 3. Notebook Sections (in detail)

---

### Section 1: Introduction to Synthetic Data Generation in Finance

This section will introduce the concept of synthetic data generation, its growing importance in the financial sector, and the specific advanced deep learning techniques (VAEs and GANs) that will be explored. It will highlight applications such as privacy preservation, stress testing, and market scenario simulation.

---

### Section 2: Setting Up the Environment and Loading Financial Data

This section will cover the necessary library imports and the loading of a sample financial time series dataset. For consistency and reproducibility, we will generate a mock dataset that mimics real financial time series, such as historical stock prices for multiple assets.

**Code Cell (Implementation): Import Libraries**
```
Import `numpy`, `pandas`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `matplotlib.pyplot`, `seaborn`, `sklearn.preprocessing`.
```

**Code Cell (Execution): Generate Mock Financial Time Series Data**
```
Call `generate_mock_financial_time_series_data` to create a synthetic dataset.
The dataset should represent daily closing prices for 5 different stocks over 252 trading days (approx. one year).
The data will be a Pandas DataFrame with columns `Asset_1_Price`, `Asset_2_Price`, ..., `Asset_5_Price`.
Each asset's price will follow a random walk with a drift, and some assets will have simulated correlation.
Set a specific random seed for reproducibility.
Display the first 5 rows and basic descriptive statistics of the generated DataFrame.
```

**Markdown Cell (Explanation): Understanding the Mock Financial Data**
The mock financial time series data simulates daily closing prices for five hypothetical assets. This type of data is common in finance for tasks like portfolio management, risk assessment, and market analysis. The generation process introduces realistic characteristics such as trends, volatility, and inter-asset correlations, making it suitable for training generative models.
The table above shows the initial values of these simulated stock prices, while the descriptive statistics provide an overview of their range, average, and variability.

---

### Section 3: Data Preprocessing for Deep Learning

Financial time series data often requires preprocessing before being fed into deep learning models. This typically includes scaling (normalization or standardization) to bring values into a similar range, which aids model convergence and performance. For sequence models, data also needs to be structured into input sequences and target sequences.

**Code Cell (Implementation): Define Data Preprocessing Function**
```
Define `preprocess_financial_data(data, sequence_length, scaler_type='minmax')`.
This function will:
1.  Initialize a `MinMaxScaler` from `sklearn.preprocessing` to scale data to the range [0, 1].
2.  Fit the scaler on the input `data` and transform it.
3.  Reshape the scaled data into sequences suitable for recurrent neural networks or transformer models. Each sequence will have `sequence_length` time steps.
4.  Return the scaled data sequences and the fitted scaler.
```

**Code Cell (Execution): Preprocess the Financial Data**
```
Set `sequence_length = 10`.
Call `preprocess_financial_data` with the generated mock financial data and the defined `sequence_length`.
Store the returned scaled data sequences and the scaler.
Print the shape of the preprocessed data.
```

**Markdown Cell (Explanation): Impact of Data Preprocessing**
Data preprocessing transforms raw financial time series into a format optimal for deep learning models. Scaling ensures that features contribute equally to the model's learning process and prevents issues caused by varying magnitudes. Reshaping converts the continuous time series into discrete sequences, allowing the models to learn temporal dependencies. For instance, a `sequence_length` of 10 means each input sample for the models will consist of 10 consecutive daily price observations for all assets.

---

### Section 4: Variational Autoencoders (VAEs) Theory

Variational Autoencoders (VAEs) are generative models that learn a probabilistic mapping from input data to a lower-dimensional latent space. Unlike traditional autoencoders, VAEs model the latent space as a probability distribution, typically Gaussian, allowing for the generation of diverse and realistic synthetic data by sampling from this learned distribution.
A VAE consists of two main components:
*   **Encoder:** Maps the input data $x$ to parameters (mean $\mu$ and log-variance $\log(\sigma^2)$) of a latent distribution $q_\phi(z|x)$.
*   **Decoder:** Reconstructs the original data $\hat{x}$ from a sample $z$ drawn from the latent distribution $p_\theta(x|z)$.

The VAE objective function maximizes the Evidence Lower Bound (ELBO), which is composed of two terms:
1.  **Reconstruction Loss:** Measures how well the decoder reconstructs the input data. For continuous data, this is often Mean Squared Error (MSE):
    $$L_{recon}(\theta; x, z) = ||x - \hat{x}||^2$$
2.  **KL Divergence Loss:** A regularization term that measures the difference between the learned latent distribution $q_\phi(z|x)$ and a prior distribution $p(z)$ (typically a standard normal distribution $N(0, I)$). This term encourages the latent space to be well-structured and continuous, enabling smooth interpolation and effective generation.
    For Gaussian latent space $q_\phi(z|x) = N(\mu_\phi(x), \Sigma_\phi(x))$ and prior $p(z) = N(0, I)$, the KL divergence is:
    $$D_{KL}(N(\mu, \Sigma) || N(0, I)) = 0.5 \sum_{i=1}^D (\sigma_i^2 + \mu_i^2 - 1 - \log(\sigma_i^2))$$
The overall VAE loss function is then a combination of these two terms, typically weighted:
$$L_{VAE}(\theta, \phi; x) = E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$
During training, a reparameterization trick $z = \mu + \sigma \cdot \epsilon$ (where $\epsilon \sim N(0, I)$) is used to enable backpropagation through the stochastic sampling process.

---

### Section 5: Implementing the VAE Architecture

This section details the construction of the VAE's Encoder and Decoder networks. For time series, these will typically involve recurrent layers to capture temporal dependencies.

**Code Cell (Implementation): Define VAE Encoder, Decoder, and Model Classes**
```
Define `VAEEncoder(torch.nn.Module)`:
-   Takes `input_dim`, `hidden_dim`, `latent_dim` as parameters.
-   Uses `torch.nn.LSTM` or `torch.nn.GRU` layers followed by `torch.nn.Linear` layers to output `mu` and `log_var`.

Define `VAEDecoder(torch.nn.Module)`:
-   Takes `latent_dim`, `hidden_dim`, `output_dim` as parameters.
-   Uses `torch.nn.Linear` layers followed by `torch.nn.LSTM` or `torch.nn.GRU` layers to reconstruct the sequence.

Define `VAEModel(torch.nn.Module)`:
-   Comprises an `encoder` and a `decoder`.
-   Includes a `reparameterize` method to sample from the latent distribution.
-   The `forward` method takes input `x`, passes it through the encoder to get `mu` and `log_var`, samples `z` using reparameterization, and passes `z` through the decoder to get `reconstruction`.
-   Returns `reconstruction`, `mu`, `log_var`.
```

**Code Cell (Execution): Instantiate the VAE Model**
```
Set `input_dim` to the number of features in the preprocessed data (e.g., 5 for 5 assets).
Set `hidden_dim` for the recurrent layers (e.g., 64).
Set `latent_dim` for the VAE's latent space (e.g., 32).
Set `output_dim` to match `input_dim`.
Instantiate `VAEModel` with these parameters.
Move the model to the appropriate device (CPU/GPU).
Print the VAE model structure.
```

**Markdown Cell (Explanation): VAE Model Architecture**
The VAE model is built with an encoder and a decoder. The encoder processes the input financial time series using recurrent layers (like LSTMs or GRUs) to capture sequence information, then maps this to the mean and log-variance of a latent Gaussian distribution. The reparameterization trick allows us to sample from this distribution while maintaining differentiability for backpropagation. The decoder, also using recurrent layers, takes this latent sample and reconstructs the original financial time series. This architecture allows the VAE to learn a compressed, probabilistic representation of the data.

---

### Section 6: VAE Loss Function and Training Loop

The VAE is trained by minimizing a combined loss function that balances reconstruction accuracy with regularization of the latent space. The training loop iteratively feeds mini-batches of data to the VAE, calculates the loss, and updates the model's parameters.

**Code Cell (Implementation): Define VAE Loss and Training Epoch Functions**
```
Define `vae_loss_function(reconstruction, x, mu, log_var, beta=1.0)`:
-   Calculates the Reconstruction Loss using `torch.nn.functional.mse_loss` between `reconstruction` and `x`.
-   Calculates the KL Divergence Loss using the formula: `0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var)`.
-   Returns the sum of Reconstruction Loss and `beta * KL_Divergence_Loss`.

Define `train_vae_epoch(model, dataloader, optimizer, device, beta)`:
-   Sets the model to training mode.
-   Iterates through the `dataloader`.
-   For each batch:
    -   Moves batch data to `device`.
    -   Performs forward pass to get `reconstruction`, `mu`, `log_var`.
    -   Calculates `loss` using `vae_loss_function`.
    -   Performs backward pass and `optimizer.step()`.
    -   Returns average `loss` for the epoch.
```

**Code Cell (Execution): Instantiate VAE Optimizer and Loss Function**
```
Define `learning_rate = 1e-3`.
Instantiate `torch.optim.Adam` optimizer for the VAE model.
Define `beta = 0.1` for the KL divergence weight.
```

**Markdown Cell (Explanation): VAE Training Mechanics**
The VAE loss function is crucial for balancing reconstruction quality with the desired structure of the latent space. The `beta` parameter allows for tuning the influence of the KL divergence term; a higher `beta` encourages a latent space closer to a standard normal distribution, while a lower `beta` prioritizes reconstruction accuracy. The `Adam` optimizer is used to efficiently adjust the model's weights during training, aiming to minimize this combined loss function.

---

### Section 7: Training the VAE Model

The training process involves iterating over the dataset for a specified number of epochs, applying the `train_vae_epoch` function for each epoch.

**Code Cell (Implementation): Define VAE Training Orchestration Function**
```
Define `train_vae_model(model, train_dataloader, epochs, optimizer, device, beta)`:
-   Initializes empty lists to store `reconstruction_losses` and `kl_divergence_losses` per epoch.
-   Loops for `epochs`:
    -   Calls `train_vae_epoch`.
    -   (Optional: Adds logic to calculate and store separate reconstruction and KL divergence components for plotting).
    -   Prints epoch number and current loss.
-   Returns `reconstruction_losses` and `kl_divergence_losses`.
```

**Code Cell (Execution): Execute VAE Training**
```
Set `epochs = 50`.
Set `batch_size = 32`.
Create `TensorDataset` and `DataLoader` from the preprocessed data.
Call `train_vae_model` with the VAE model, dataloader, epochs, optimizer, device, and beta.
Store the returned loss histories.
```

**Markdown Cell (Explanation): VAE Training Process Overview**
Training the VAE involves repeatedly exposing the model to the preprocessed financial data. For each epoch, the model processes all data in mini-batches, calculates the combined reconstruction and KL divergence loss, and updates its weights. The training process aims to find model parameters that allow the encoder to map real data to a meaningful latent space and the decoder to accurately reconstruct data from samples within that space, while keeping the latent distribution close to a prior. The collected loss histories will be used to visualize the training progress.

---

### Section 8: Visualizing VAE Training Progress

Visualizing loss curves provides insights into the model's learning process, indicating convergence, overfitting, or underfitting.

**Code Cell (Implementation): Define VAE Loss Plotting Function**
```
Define `plot_vae_losses(reconstruction_losses, kl_divergence_losses, epochs)`:
-   Generates a line plot showing `reconstruction_losses` over epochs.
-   Generates a line plot showing `kl_divergence_losses` over epochs.
-   Adds appropriate titles and labels.
-   Uses `matplotlib.pyplot` and `seaborn` for aesthetics.
```

**Code Cell (Execution): Plot VAE Loss Curves**
```
Call `plot_vae_losses` with the stored `reconstruction_losses` and `kl_divergence_losses`.
```

**Markdown Cell (Explanation): Interpreting VAE Loss Curves**
The VAE loss curves illustrate how well the model is learning. A decreasing reconstruction loss indicates that the decoder is getting better at recreating the input data. A stable or decreasing KL divergence loss shows that the encoder is successfully mapping the input data to a latent distribution that is consistent with the chosen prior. Observing these curves helps determine if the model has converged and whether the balance between reconstruction and regularization is appropriate.

---

### Section 9: Generating Synthetic Data with VAE

After training, the VAE's decoder can be used to generate new synthetic financial data by sampling random vectors from the standard normal distribution (our prior for the latent space) and passing them through the decoder.

**Code Cell (Implementation): Define VAE Synthetic Data Generation Function**
```
Define `generate_synthetic_data_vae(model, num_samples, latent_dim, device, scaler, sequence_length)`:
-   Sets the model to evaluation mode.
-   Generates `num_samples` random noise vectors from a standard normal distribution of size `latent_dim` using `torch.randn`.
-   Passes these latent vectors through the VAE decoder.
-   Inverse transforms the output using the fitted `scaler` to revert to the original data scale.
-   Returns the generated synthetic data as a NumPy array or Pandas DataFrame.
```

**Code Cell (Execution): Generate Synthetic Data using VAE**
```
Set `num_synthetic_samples = 100`.
Call `generate_synthetic_data_vae` with the trained VAE model, number of samples, latent dimension, device, scaler, and sequence length.
Print the shape of the generated synthetic data.
Display the first 5 rows of the generated synthetic data (after inverse scaling).
```

**Markdown Cell (Explanation): VAE Synthetic Data Generation**
The ability to generate new, unseen data is the core purpose of a generative model. For the VAE, this process involves drawing random samples from a simple prior distribution (e.g., a standard normal distribution) in the latent space. These samples are then passed through the trained decoder, which transforms them into synthetic financial time series data that structurally resembles the real data the VAE was trained on. The inverse scaling step ensures the data is presented in its original, interpretable range.

---

### Section 10: Generative Adversarial Networks (GANs) Theory

Generative Adversarial Networks (GANs) employ a game-theoretic approach to synthetic data generation. They consist of two competing neural networks:
*   **Generator (G):** Takes random noise $z$ as input and attempts to produce synthetic data $G(z)$ that mimics the real data distribution $p_{data}(x)$.
*   **Discriminator (D):** Takes either real data $x$ or synthetic data $G(z)$ as input and tries to distinguish between the two, outputting a probability that the input is real.

The two networks are trained simultaneously in a minimax game:
*   The **Discriminator** is trained to maximize its ability to correctly classify real vs. fake data.
*   The **Generator** is trained to minimize the Discriminator's ability to distinguish its synthetic outputs from real data, effectively trying to "fool" the discriminator.

The value function for this minimax game is given by:
$$\min_G \max_D V(D, G) = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_z(z)}[\log (1 - D(G(z)))]$$
Here, $D(x)$ is the Discriminator's output for real data $x$, and $D(G(z))$ is its output for synthetic data $G(z)$. The Generator aims to make $D(G(z))$ close to 1 (meaning the Discriminator thinks it's real), while the Discriminator aims to make $D(x)$ close to 1 and $D(G(z))$ close to 0. When the GAN converges, the Generator produces data that is indistinguishable from real data, and the Discriminator outputs $0.5$ for both real and synthetic inputs.

---

### Section 11: Implementing the GAN Architecture

This section describes the neural network architectures for the GAN's Generator and Discriminator, also leveraging recurrent layers for financial time series data.

**Code Cell (Implementation): Define GAN Generator and Discriminator Classes**
```
Define `GANGenerator(torch.nn.Module)`:
-   Takes `latent_dim`, `hidden_dim`, `output_dim`, `sequence_length` as parameters.
-   Uses `torch.nn.Linear` layers to project latent noise to an initial sequence representation.
-   Uses `torch.nn.LSTM` or `torch.nn.GRU` layers to generate the synthetic time series.
-   Applies `torch.nn.Tanh` activation to ensure output is within a specific range (e.g., [-1, 1], matching preprocessed data if using `StandardScaler`).

Define `GANDiscriminator(torch.nn.Module)`:
-   Takes `input_dim`, `hidden_dim`, `sequence_length` as parameters.
-   Uses `torch.nn.LSTM` or `torch.nn.GRU` layers to process the input sequence.
-   Uses `torch.nn.Linear` layers to output a single probability score (e.g., via `torch.nn.Sigmoid` for binary classification).
```

**Code Cell (Execution): Instantiate the GAN Models**
```
Set `latent_dim` for the GAN's input noise (e.g., 64).
Set `hidden_dim` for recurrent layers (e.g., 128).
Set `output_dim` to match the number of features (e.g., 5).
Set `sequence_length` (e.g., 10).
Instantiate `GANGenerator` and `GANDiscriminator` with these parameters.
Move both models to the appropriate device (CPU/GPU).
Print the Generator and Discriminator model structures.
```

**Markdown Cell (Explanation): GAN Model Architectures**
The GAN comprises two distinct neural networks: the Generator and the Discriminator. The Generator takes a random noise vector as input and transforms it through linear and recurrent layers to produce a synthetic financial time series. The Discriminator takes a time series (either real or generated) and, using similar recurrent and linear layers, outputs a single value representing the probability that the input is real. The `Tanh` activation in the Generator ensures the generated data is within a specific range, which is critical when matching normalized real data.

---

### Section 12: GAN Loss Functions and Training Loop

The GAN training process involves an alternating optimization strategy, updating the Discriminator and Generator sequentially using their respective loss functions.

**Code Cell (Implementation): Define GAN Loss and Training Epoch Functions**
```
Define `gan_discriminator_loss(real_output, fake_output, loss_fn)`:
-   Calculates loss for real data: `loss_fn(real_output, torch.ones_like(real_output))`.
-   Calculates loss for fake data: `loss_fn(fake_output, torch.zeros_like(fake_output))`.
-   Returns the sum of these two losses.

Define `gan_generator_loss(fake_output, loss_fn)`:
-   Calculates loss for generator: `loss_fn(fake_output, torch.ones_like(fake_output))`.
-   Returns this loss.

Define `train_gan_epoch(generator, discriminator, dataloader, gen_optimizer, disc_optimizer, loss_fn, device, latent_dim)`:
-   Sets models to training mode.
-   Iterates through `dataloader`.
-   For each batch:
    -   **Train Discriminator:**
        -   Gets real data from batch, moves to device.
        -   Generates fake data using `generator` from random noise.
        -   Calculates `real_output` and `fake_output` from `discriminator`.
        -   Calculates `disc_loss` using `gan_discriminator_loss`.
        -   Performs backward pass and `disc_optimizer.step()`.
    -   **Train Generator:**
        -   Generates new fake data from new random noise.
        -   Calculates `fake_output` from `discriminator`.
        -   Calculates `gen_loss` using `gan_generator_loss`.
        -   Performs backward pass and `gen_optimizer.step()`.
-   Returns average `disc_loss` and `gen_loss` for the epoch.
```

**Code Cell (Execution): Instantiate GAN Optimizers and Loss Function**
```
Define `gen_learning_rate = 2e-4` and `disc_learning_rate = 2e-4`.
Instantiate `torch.optim.Adam` for the `generator` and `discriminator` separately.
Define `loss_fn = torch.nn.BCELoss()` (Binary Cross-Entropy Loss).
```

**Markdown Cell (Explanation): GAN Training Mechanics**
GAN training is a delicate dance between the Generator and Discriminator. The `BCELoss` is ideal for their binary classification tasks. The Discriminator's goal is to accurately classify real data as 1 and fake data as 0, while the Generator's goal is to produce data that the Discriminator classifies as 1. The separate optimizers allow independent updates, reflecting their adversarial roles. This adversarial process drives both networks to improve, with the Generator striving to produce increasingly realistic data.

---

### Section 13: Training the GAN Model

This section details the GAN training loop over multiple epochs, invoking the `train_gan_epoch` function.

**Code Cell (Implementation): Define GAN Training Orchestration Function**
```
Define `train_gan_model(generator, discriminator, train_dataloader, epochs, gen_optimizer, disc_optimizer, loss_fn, device, latent_dim)`:
-   Initializes empty lists to store `generator_losses` and `discriminator_losses` per epoch.
-   Loops for `epochs`:
    -   Calls `train_gan_epoch`.
    -   Stores the returned `disc_loss` and `gen_loss`.
    -   Prints epoch number and current losses.
-   Returns `generator_losses` and `discriminator_losses`.
```

**Code Cell (Execution): Execute GAN Training**
```
Set `epochs = 50`.
Set `batch_size = 32`.
Create `TensorDataset` and `DataLoader` from the preprocessed data.
Call `train_gan_model` with the generator, discriminator, dataloader, epochs, optimizers, loss function, device, and latent dimension.
Store the returned loss histories.
```

**Markdown Cell (Explanation): GAN Training Process Overview**
Training a GAN requires careful orchestration of the Generator and Discriminator updates. In each epoch, both networks are updated, often with the Discriminator being trained slightly more or with specific hyperparameters to ensure it remains a credible adversary. This simplified training loop focuses on demonstrating the core adversarial learning process, where the Generator learns to map random noise to data that can fool the Discriminator, while the Discriminator continuously improves its ability to detect synthetic data.

---

### Section 14: Visualizing GAN Training Progress

Plotting the Generator and Discriminator losses helps monitor the adversarial training process and identify potential issues like mode collapse or training instability.

**Code Cell (Implementation): Define GAN Loss Plotting Function**
```
Define `plot_gan_losses(generator_losses, discriminator_losses, epochs)`:
-   Generates a line plot showing `generator_losses` over epochs.
-   Generates a line plot showing `discriminator_losses` over epochs.
-   Adds appropriate titles and labels.
-   Uses `matplotlib.pyplot` and `seaborn` for aesthetics.
```

**Code Cell (Execution): Plot GAN Loss Curves**
```
Call `plot_gan_losses` with the stored `generator_losses` and `discriminator_losses`.
```

**Markdown Cell (Explanation): Interpreting GAN Loss Curves**
Interpreting GAN loss curves is different from standard supervised learning. Ideally, both the Generator and Discriminator losses should fluctuate and eventually stabilize around a certain point, indicating a successful adversarial training equilibrium. If the Generator loss drops significantly while the Discriminator loss remains high, it might suggest the Generator has "won" too easily, potentially leading to mode collapse (generating limited diversity). Conversely, if the Discriminator loss drops to near zero, it might mean the Generator is not improving, and the Discriminator can easily distinguish fake data.

---

### Section 15: Generating Synthetic Data with GAN

Once the GAN is trained, the Generator can produce novel synthetic financial time series simply by transforming random noise.

**Code Cell (Implementation): Define GAN Synthetic Data Generation Function**
```
Define `generate_synthetic_data_gan(generator, num_samples, latent_dim, device, scaler, sequence_length)`:
-   Sets the generator to evaluation mode.
-   Generates `num_samples` random noise vectors from a standard normal distribution of size `latent_dim` using `torch.randn`.
-   Passes these noise vectors through the `generator`.
-   Inverse transforms the output using the fitted `scaler` to revert to the original data scale.
-   Returns the generated synthetic data as a NumPy array or Pandas DataFrame.
```

**Code Cell (Execution): Generate Synthetic Data using GAN**
```
Set `num_synthetic_samples = 100`.
Call `generate_synthetic_data_gan` with the trained GAN generator, number of samples, latent dimension, device, scaler, and sequence length.
Print the shape of the generated synthetic data.
Display the first 5 rows of the generated synthetic data (after inverse scaling).
```

**Markdown Cell (Explanation): GAN Synthetic Data Generation**
Generating synthetic data with a GAN involves feeding random noise into the trained Generator network. The Generator, having learned to transform noise into data that can fool the Discriminator, will produce synthetic financial time series that visually and statistically resemble the real data it was trained on. This demonstrates the Generator's capability to learn and replicate complex data distributions from simple random inputs. The inverse scaling ensures the output is in the original price range.

---

### Section 16: Comparative Analysis: Data Distributions (Histograms/KDEs)

A critical step in validating synthetic data is to compare its statistical properties with those of the original real data. Visualizing feature distributions helps assess how well the models capture the underlying data characteristics.

**Code Cell (Implementation): Define Feature Distribution Plotting Function**
```
Define `plot_feature_distributions(real_data, vae_synthetic_data, gan_synthetic_data, feature_names)`:
-   Creates a figure with subplots, one for each feature in `feature_names`.
-   For each feature:
    -   Plots the histogram/KDE of the `real_data` feature using `seaborn.histplot(kde=True)`.
    -   Overlays the histogram/KDE of the `vae_synthetic_data` feature.
    -   Overlays the histogram/KDE of the `gan_synthetic_data` feature.
    -   Adds legend, title (e.g., "Distribution of [Feature Name]"), and labels.
-   Uses `matplotlib.pyplot` for layout and display.
```

**Code Cell (Execution): Plot Feature Distributions**
```
Extract a subset of the original real data and the generated synthetic data (VAE and GAN) to compare their features.
Call `plot_feature_distributions` with the relevant real, VAE synthetic, and GAN synthetic data (e.g., final price point of each sequence or aggregated statistics like mean/variance of sequences) and the corresponding feature names (e.g., `Asset_1_Price_End`, `Asset_2_Price_End`).
```

**Markdown Cell (Explanation): Comparing Data Distributions**
These distribution plots allow for a visual assessment of how closely the synthetic data generated by VAEs and GANs matches the statistical properties of the real financial data. Overlapping histograms or KDEs indicate that the synthetic data successfully replicates the real data's shape, spread, and central tendency for individual features. Discrepancies might highlight areas where the generative models could be improved, or limitations in their ability to capture certain nuances of the real data distribution.

---

### Section 17: Comparative Analysis: Time Series Visualizations

For time series data, it's essential to visually inspect the generated sequences against real ones to evaluate their realism, temporal dependencies, and overall structure.

**Code Cell (Implementation): Define Time Series Plotting Function**
```
Define `plot_synthetic_and_real_time_series(real_data_sequences, vae_synthetic_sequences, gan_synthetic_sequences, num_plots=5, asset_index=0)`:
-   Creates a figure with `num_plots` subplots.
-   For each subplot:
    -   Selects a random real time series for `asset_index`.
    -   Selects a random VAE synthetic time series for `asset_index`.
    -   Selects a random GAN synthetic time series for `asset_index`.
    -   Plots these three time series on the same subplot.
    -   Adds legend, title (e.g., "Real vs. Synthetic Time Series (Asset X)"), and labels.
-   Uses `matplotlib.pyplot` for layout and display.
```

**Code Cell (Execution): Plot Sample Time Series**
```
Call `plot_synthetic_and_real_time_series` with the original preprocessed real data sequences, VAE synthetic sequences, and GAN synthetic sequences, specifying `num_plots=5` and `asset_index=0` (e.g., for Asset 1).
```

**Markdown Cell (Explanation): Visual Assessment of Time Series Realism**
Visual inspection of time series plots provides an intuitive understanding of the generative models' performance. Realistic synthetic time series should exhibit similar trends, volatility patterns, and overall shape as the real financial data. This comparison helps identify if the models are capturing the underlying temporal dynamics, or if the generated data appears noisy, overly smooth, or lacks key characteristics found in real market movements.

---

### Section 18: Comparative Analysis: Quantitative Metrics

Beyond visual comparisons, quantitative metrics offer a more rigorous assessment of synthetic data quality. Statistical tests can compare distributions, while simple metrics like mean and variance provide a quick numerical overview.

**Code Cell (Implementation): Define Statistical Metrics and K-S Test Functions**
```
Define `calculate_statistical_metrics(data_dict, feature_names)`:
-   Takes a dictionary mapping data labels (e.g., 'Real', 'VAE', 'GAN') to data arrays/DataFrames.
-   Calculates mean and standard deviation for each feature in `feature_names` for each dataset.
-   Prints these statistics in a clear, tabular format.

Define `perform_kolmogorov_smirnov_test(real_data, synthetic_data, feature_names)`:
-   Iterates through `feature_names`.
-   For each feature, performs a Kolmogorov-Smirnov (K-S) test between the `real_data` feature and the `synthetic_data` feature using `scipy.stats.ks_2samp`.
-   Prints the K-S statistic and p-value for each comparison.
-   Explains the interpretation of p-values (e.g., high p-value suggests failure to reject null hypothesis that distributions are similar).
```

**Code Cell (Execution): Calculate and Display Quantitative Metrics**
```
Create a dictionary `data_to_compare` containing the real, VAE synthetic, and GAN synthetic data (e.g., the flattened data or specific summary statistics per sequence).
Call `calculate_statistical_metrics` with this dictionary and relevant `feature_names`.
Call `perform_kolmogorov_smirnov_test` to compare real data with VAE synthetic data for each feature.
Call `perform_kolmogorov_smirnov_test` to compare real data with GAN synthetic data for each feature.
```

**Markdown Cell (Explanation): Interpreting Quantitative Comparisons**
Quantitative metrics provide a numerical basis for comparing real and synthetic data.
*   **Mean and Standard Deviation:** These basic statistics offer a first glance at whether the synthetic data preserves the central tendency and variability of the real data.
*   **Kolmogorov-Smirnov (K-S) Test:** This non-parametric test quantifies the difference between the empirical cumulative distribution functions of two samples. A high p-value (typically $p > 0.05$) suggests that we cannot reject the null hypothesis that the two samples are drawn from the same underlying distribution, indicating good statistical similarity between the real and synthetic data. Lower p-values indicate a significant difference.

These metrics complement visual analysis, offering a more objective measure of how well VAEs and GANs capture the statistical essence of financial data.

---

### Section 19: Adjusting Latent Variables (VAE Focus)

One of the strengths of VAEs is their interpretable latent space. By systematically varying specific dimensions of a latent vector, we can observe how these changes influence the characteristics of the generated synthetic data. This demonstrates the VAE's ability to learn disentangled representations of the data.

**Code Cell (Implementation): Define Function for Generating Data with Adjusted Latent Variables**
```
Define `generate_with_adjusted_latent(model, original_latent_sample, latent_dim_to_adjust, adjustment_range, num_steps, device, scaler, sequence_length)`:
-   Takes an `original_latent_sample` (e.g., mean of latent space or a random sample).
-   Creates multiple copies of this `original_latent_sample`.
-   For each copy, perturbs the `latent_dim_to_adjust`-th dimension across `num_steps` within the `adjustment_range`.
-   Passes these adjusted latent vectors through the VAE decoder.
-   Inverse transforms the output using the `scaler`.
-   Returns a list of generated synthetic time series, corresponding to each adjustment step.
```

**Code Cell (Execution): Generate and Plot Synthetic Data with Latent Variable Adjustments**
```
Generate a reference `original_latent_sample` (e.g., a zero vector or the mean of the latent space from the trained VAE).
Choose `latent_dim_to_adjust = 0` (the first latent dimension).
Set `adjustment_range = (-3, 3)` and `num_steps = 10`.
Call `generate_with_adjusted_latent` to get synthetic data for varying the chosen latent dimension.
Plot these generated time series, showing how changing one latent variable affects the output. For simplicity, plot only Asset 1's price.
```

**Markdown Cell (Explanation): Impact of Latent Variable Manipulation**
Manipulating the VAE's latent variables offers a powerful way to understand the underlying factors the model has learned. By systematically changing one dimension of the latent vector while keeping others constant, we can observe how a specific "feature" of the synthetic data changes. For financial data, this could correspond to underlying market factors like volatility, trend, or correlation. This interactive exploration highlights the VAE's capacity for controlled data generation and could be useful for scenario analysis or generating data with specific characteristics.

---

### Section 20: Conclusion and Financial Applications

This section summarizes the key findings from implementing and comparing VAEs and GANs for synthetic financial data generation. It reiterates their practical applications in finance and discusses future potential.

---
```
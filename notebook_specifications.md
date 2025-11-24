
# Technical Specification for Jupyter Notebook: SynthFin Data Generator: VAEs & GANs

## 1. Notebook Overview

### Learning Goals
Upon completing this notebook, Financial Data Engineers will be able to:
1.  Understand the theoretical foundations and practical applications of Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) for synthetic financial data generation.
2.  Implement VAE and GAN architectures using a deep learning framework.
3.  Train VAE and GAN models on anonymized financial datasets, monitoring key training metrics like loss curves.
4.  Generate synthetic financial data from trained VAE and GAN models.
5.  Perform comparative statistical analysis between real and synthetic financial data, evaluating quality and fidelity.
6.  Explore the impact of adjusting latent variables or input noise on synthetic data generation to simulate various financial scenarios.
7.  Appreciate the financial applications of synthetic data, including privacy preservation, stress-testing, and market scenario simulation.

### Who the Notebook is Targeted To
This notebook is designed for **Financial Data Engineers** who are looking to:
*   Enhance their understanding of advanced deep learning techniques (VAEs and GANs) for synthetic data generation.
*   Acquire practical skills in implementing, training, and evaluating generative models in a financial context.
*   Explore methods for creating privacy-preserving and robust datasets for model development and validation.
*   Visually and statistically analyze the fidelity of generated synthetic financial data.

## 2. Code Requirements

### List of Expected Libraries
*   `numpy`: For numerical operations and synthetic data generation.
*   `pandas`: For data manipulation and tabular data handling.
*   `matplotlib.pyplot`: For basic plotting and visualization.
*   `seaborn`: For enhanced statistical data visualization.
*   `sklearn.preprocessing.MinMaxScaler`: For data scaling.
*   `tensorflow`: For building and training deep learning models (VAEs and GANs). Specifically, `tensorflow.keras` will be utilized for its high-level API.
*   `scipy.stats.gaussian_kde`: For Kernel Density Estimation plots in distribution comparisons.

### List of Algorithms or Functions to be Implemented
*   `generate_synthetic_financial_data(num_samples, random_seed)`: Creates a synthetic tabular financial dataset with specified features and plausible statistical distributions.
*   `load_financial_data(dataframe)`: A placeholder function to "load" the pre-generated synthetic financial data, mimicking real-world data import.
*   `preprocess_data(dataframe)`: Scales numerical features of the input DataFrame using `MinMaxScaler`, returning the scaled data and the fitted scaler object.
*   `build_vae(input_dim, latent_dim, hidden_dim_encoder, hidden_dim_decoder)`: Constructs the VAE Encoder, Decoder, and the full VAE model (using TensorFlow Keras subclassing or functional API), encapsulating the reparameterization trick.
*   `train_vae_model(vae_model, train_data, epochs, batch_size, learning_rate)`: Compiles the VAE model with a custom VAE loss function (combining reconstruction and KL divergence) and an Adam optimizer, then trains it on the provided data, returning the training history.
*   `plot_vae_loss(history)`: Generates a line plot showing VAE reconstruction loss, KL divergence loss, and total VAE loss over training epochs.
*   `generate_vae_synthetic_data(decoder_model, num_samples, latent_dim, scaler)`: Generates synthetic data by sampling from a standard normal distribution, passing through the VAE decoder, and inverse-transforming the output using the provided `MinMaxScaler`.
*   `build_gan(input_dim, latent_dim, hidden_dim_generator, hidden_dim_discriminator)`: Constructs the GAN Generator, Discriminator, and the combined GAN model (for generator training), using TensorFlow Keras.
*   `train_gan_model(generator, discriminator, gan_model, train_data, epochs, batch_size, latent_dim, learning_rate)`: Implements the adversarial training loop for the GAN, alternating between training the Discriminator and the Generator, returning a history of their respective losses.
*   `plot_gan_loss(history)`: Generates a line plot showing Generator loss and Discriminator loss over training epochs.
*   `generate_gan_synthetic_data(generator_model, num_samples, latent_dim, scaler)`: Generates synthetic data by sampling random noise, passing it through the trained GAN generator, and inverse-transforming the output.
*   `compare_statistics(real_data, synthetic_data_vae, synthetic_data_gan)`: Computes and displays comprehensive descriptive statistics (mean, std dev, min, max, quartiles) for all features across real, VAE synthetic, and GAN synthetic datasets.
*   `compare_distributions(real_data, synthetic_data_vae, synthetic_data_gan, feature_names)`: Generates overlaid histogram and Kernel Density Estimate (KDE) plots for each financial feature, allowing visual comparison of distributions between real and synthetic data.
*   `plot_time_series_comparison(real_data, synthetic_data_vae, synthetic_data_gan, feature_name, num_samples_to_plot)`: Generates line plots of selected features over a sample index (treating synthetic samples as a sequence) to qualitatively compare visual patterns between real and synthetic data.
*   `explore_vae_latent_space(decoder_model, latent_dim, scaler, original_data_stats, num_variations, feature_to_vary)`: Generates and analyzes variations in synthetic data by incrementally perturbing a specific dimension of the VAE's latent vector.
*   `explore_gan_noise_variation(generator_model, latent_dim, scaler, original_data_stats, num_variations, noise_factor_range)`: Generates and analyzes variations in synthetic data by scaling the GAN's input noise vector by different factors.

### Visualization Like Charts, Tables, Plots That Should Be Generated
*   **Data Inspection Tables**: `pandas.DataFrame.head()` and `pandas.DataFrame.describe()` outputs for original and scaled financial data.
*   **VAE Training Loss Plot**: A multi-line plot displaying VAE reconstruction loss, KL divergence loss, and total VAE loss against training epochs.
*   **GAN Training Loss Plot**: A multi-line plot displaying Generator loss and Discriminator loss against training epochs.
*   **Feature Distribution Comparison Plots**: For each financial feature, an overlaid plot featuring histograms and Kernel Density Estimates (KDEs) of the real data, VAE synthetic data, and GAN synthetic data.
*   **Statistical Summary Tables**: Consolidated tables comparing descriptive statistics (e.g., mean, standard deviation, min, max, quartiles) for each feature across the real, VAE synthetic, and GAN synthetic datasets.
*   **Time Series-like Plots**: Line plots for selected features (e.g., `Daily_Return`, `Volatility`) showing the first `N` real data points and `N` generated synthetic data points (treating sample index as a pseudo-time axis for synthetic data) for visual pattern comparison.
*   **Latent Space/Noise Variation Plots**: Visual representations (e.g., line plots of feature means or distributions) illustrating how adjustments to VAE latent vectors or GAN input noise impact the generated synthetic data's characteristics.

## 3. Notebook Sections (in detail)

### Section 1: Introduction to Synthetic Financial Data Generation

*   **Markdown Cell:**
    ```markdown
    # SynthFin Data Generator: VAEs & GANs for Financial Data Synthesis

    Welcome to the SynthFin Data Generator notebook! In the dynamic world of finance, data is paramount, but often comes with constraints like privacy concerns, limited availability, or the need to simulate extreme, rare events. **Synthetic data generation** addresses these challenges by creating artificial datasets that mimic the statistical properties of real data without exposing sensitive information.

    This notebook explores two advanced deep learning techniques for synthetic data generation: **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**. As Financial Data Engineers, understanding and applying these models is crucial for tasks such as:
    *   **Privacy-preserving data sharing:** Creating anonymized datasets for collaboration or external research.
    *   **Stress-testing financial models:** Generating data for scenarios not observed in historical records.
    *   **Simulating complex market scenarios:** Exploring various market conditions to test strategies or risk management.
    *   **Augmenting scarce datasets:** Expanding limited real datasets for more robust model training.

    We will dive into the theoretical underpinnings of VAEs and GANs, implement their architectures, train them on a simulated financial dataset, generate synthetic data, and rigorously compare the statistical properties of the generated data against the real data.
    ```

### Section 2: Setting Up the Environment and Generating Financial Data

*   **Markdown Cell:**
    ```markdown
    To begin, we set up our Python environment by importing the necessary libraries. For this notebook, we will use a synthetically generated financial dataset to ensure reproducibility and focus on the generative models. This dataset simulates anonymized daily financial features, reflecting the kind of small, tabular data often encountered by Financial Data Engineers.

    Our synthetic dataset will consist of `num_samples` rows, each representing a daily observation, and five features:
    *   `Daily_Return`: Simulates daily percentage change in an asset price.
    *   `Volatility`: Simulates rolling volatility of an asset price.
    *   `Volume_Change`: Simulates daily percentage change in trading volume.
    *   `Interest_Rate_Change`: Simulates daily change in a relevant interest rate.
    *   `Market_Sentiment_Score`: Simulates a daily sentiment score (e.g., derived from news or social media).
    ```
*   **Code Cell (Function):**
    ```python
    def generate_synthetic_financial_data(num_samples, random_seed=42):
        """
        Generates a synthetic tabular financial dataset with specified features.

        Args:
            num_samples (int): The number of samples (rows) to generate.
            random_seed (int): Seed for random number generation for reproducibility.

        Returns:
            pandas.DataFrame: A DataFrame containing the synthetic financial data.
        """
        np.random.seed(random_seed)
        
        # Generate Daily_Return: Centered around a small positive mean, slightly skewed
        daily_return = np.random.normal(loc=0.0005, scale=0.015, size=num_samples)
        
        # Generate Volatility: Positive, typically low values, some higher spikes
        # Using a log-normal distribution for positive, right-skewed values
        volatility = np.exp(np.random.normal(loc=-3.0, scale=0.8, size=num_samples))
        volatility = np.clip(volatility, 0.001, 0.1) # Ensure within plausible financial range

        # Generate Volume_Change: Centered around zero with some fluctuations
        volume_change = np.random.normal(loc=0.0, scale=0.1, size=num_samples)
        
        # Generate Interest_Rate_Change: Small changes around zero
        interest_rate_change = np.random.normal(loc=0.0, scale=0.001, size=num_samples)
        
        # Generate Market_Sentiment_Score: Bounded between -1 and 1
        market_sentiment_score = np.random.uniform(low=-0.8, high=0.8, size=num_samples)
        
        data = {
            'Daily_Return': daily_return,
            'Volatility': volatility,
            'Volume_Change': volume_change,
            'Interest_Rate_Change': interest_rate_change,
            'Market_Sentiment_Score': market_sentiment_score
        }
        
        return pd.DataFrame(data)

    def load_financial_data(dataframe):
        """
        Simulates loading financial data. In this notebook, it returns the pre-generated DataFrame.

        Args:
            dataframe (pandas.DataFrame): The pre-generated DataFrame to "load".

        Returns:
            pandas.DataFrame: The input DataFrame.
        """
        return dataframe
    ```
*   **Code Cell (Execution):**
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from scipy.stats import gaussian_kde

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Generate 1000 samples of synthetic financial data
    num_samples = 1000
    financial_df_raw = generate_synthetic_financial_data(num_samples)

    # Load the generated data (simulating loading from a file)
    financial_data = load_financial_data(financial_df_raw)

    print("First 5 rows of the raw financial dataset:")
    print(financial_data.head())

    print("\nDescriptive statistics of the raw financial dataset:")
    print(financial_data.describe())
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The synthetic financial dataset has been successfully generated and "loaded". We can observe the structure of the data and its basic statistical properties. Each row represents a daily observation, and the columns represent different financial indicators. These features are designed to mimic real-world financial data characteristics, such as `Daily_Return` often hovering around zero with varying `Volatility`, and `Market_Sentiment_Score` bounded within a specific range. The `describe()` output provides initial insights into the mean, standard deviation, and range of each feature, which are important benchmarks for evaluating our synthetic data later.
    ```

### Section 3: Data Preprocessing and Feature Scaling

*   **Markdown Cell:**
    ```markdown
    Before feeding data into neural networks like VAEs and GANs, it's crucial to preprocess it. Neural networks are sensitive to the scale and distribution of input data. **Feature scaling** helps stabilize training and improves model performance. Here, we'll apply `MinMaxScaler`, which transforms features by scaling each feature to a given range, typically $ [0, 1] $. This is particularly suitable for generative models whose output activations (like `sigmoid`) naturally produce values in this range.

    The scaling transformation for a feature $ X $ is defined as:
    $$ X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}} $$
    where $ X_{min} $ is the minimum value of the feature in the training data, and $ X_{max} $ is the maximum value. It is critical to store the `MinMaxScaler` instance used during training to **inverse transform** the generated synthetic data back to its original scale for meaningful interpretation.
    ```
*   **Code Cell (Function):**
    ```python
    def preprocess_data(dataframe):
        """
        Preprocesses the data by applying MinMaxScaler to all numerical columns.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame to scale.

        Returns:
            tuple: A tuple containing:
                - pandas.DataFrame: The scaled DataFrame.
                - sklearn.preprocessing.MinMaxScaler: The fitted scaler object.
        """
        scaler = MinMaxScaler()
        scaled_data_array = scaler.fit_transform(dataframe)
        scaled_dataframe = pd.DataFrame(scaled_data_array, columns=dataframe.columns)
        return scaled_dataframe, scaler
    ```
*   **Code Cell (Execution):**
    ```python
    # Preprocess the financial data
    scaled_financial_data, scaler = preprocess_data(financial_data)

    print("First 5 rows of the scaled financial dataset:")
    print(scaled_financial_data.head())

    print("\nDescriptive statistics of the scaled financial dataset:")
    print(scaled_financial_data.describe())
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The financial data has now been successfully scaled to the $ [0, 1] $ range. Observing the descriptive statistics, all features now fall within this interval, with minimums at 0 and maximums at 1, as expected. This standardized input is critical for the stable and efficient training of our deep learning models. The `scaler` object is saved, which will be essential later for converting the generated synthetic data back to its original financial units, making it interpretable.
    ```

### Section 4: Introduction to Variational Autoencoders (VAEs)

*   **Markdown Cell:**
    ```markdown
    **Variational Autoencoders (VAEs)** are a powerful class of generative models that combine deep learning with Bayesian inference. Unlike traditional Autoencoders (AEs) which learn a deterministic latent representation, VAEs learn a *probability distribution* over the latent space. This probabilistic approach is key to their generative capabilities.

    A VAE consists of two main parts:
    1.  **Encoder ($ q_{\phi}(\mathbf{z}|\mathbf{x}) $):** This neural network takes an input data point $ \mathbf{x} $ and outputs the parameters (mean $ \mu $ and log-variance $ \log \sigma^2 $) of a conditional probability distribution, typically a Gaussian, in the latent space. That is, $ q_{\phi}(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \mu(\mathbf{x}), \sigma^2(\mathbf{x})) $.
    2.  **Decoder ($ p_{\theta}(\mathbf{x}|\mathbf{z}) $):** This neural network takes a sample $ \mathbf{z} $ from the latent distribution and reconstructs the input data $ \mathbf{x} $.

    To make the sampling process from the latent distribution differentiable (required for backpropagation), VAEs employ the **reparameterization trick**. Instead of directly sampling $ \mathbf{z} \sim \mathcal{N}(\mu, \sigma^2) $, we sample a random noise vector $ \epsilon \sim \mathcal{N}(0, \mathbf{I}) $ and compute the latent variable as:
    $$ \mathbf{z} = \mu + \sigma \odot \epsilon $$
    where $ \odot $ denotes element-wise multiplication.

    The VAE is trained to optimize a **loss function** that balances two objectives:
    *   **Reconstruction Loss:** Measures how well the decoder reconstructs the original input data. This is typically the Mean Squared Error (MSE) for continuous data: $ -E_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] \approx \text{MSE}(\mathbf{x}, \text{decoder}(\mathbf{z})) $.
    *   **KL Divergence Loss:** A regularization term that measures the difference between the learned latent distribution $ q_{\phi}(\mathbf{z}|\mathbf{x}) $ and a predefined prior distribution $ p(\mathbf{z}) $, usually a standard normal distribution $ \mathcal{N}(0, \mathbf{I}) $. This term encourages the latent space to be well-structured and continuous, allowing for smooth data generation. For Gaussian latent space and a standard Gaussian prior, the KL divergence has a closed-form solution:
        $$ D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) = 0.5 \sum_{i=1}^{latent\_dim} (\exp(\log \sigma_i^2) + \mu_i^2 - 1 - \log \sigma_i^2) $$
    The total VAE loss is the sum of these two components:
    $$ \mathcal{L}(\theta, \phi; \mathbf{x}) = \underbrace{-E_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))}_{\text{KL Divergence Loss}} $$
    By minimizing this loss, VAEs learn to encode data into a meaningful latent space from which realistic and diverse synthetic samples can be generated.
    ```

### Section 5: Defining the VAE Architecture

*   **Markdown Cell:**
    ```markdown
    We will now define the architecture for our VAE using TensorFlow Keras. The VAE will consist of an Encoder network that maps the input financial data to parameters of a Gaussian distribution in a lower-dimensional latent space, and a Decoder network that samples from this latent space to reconstruct the financial data.

    For our VAE, we'll use `tf.keras.layers.Dense` layers (fully connected layers) with `relu` activation functions for the intermediate layers, providing non-linearity. The final layer of the decoder will use a `sigmoid` activation to ensure the output data is scaled between $ [0, 1] $, consistent with our preprocessed input data.
    ```
*   **Code Cell (Function):**
    ```python
    class Encoder(tf.keras.layers.Layer):
        def __init__(self, latent_dim, hidden_dim, name="encoder", **kwargs):
            super().__init__(name=name, **kwargs)
            self.dense_proj = tf.keras.layers.Dense(hidden_dim, activation="relu")
            self.dense_mean = tf.keras.layers.Dense(latent_dim)
            self.dense_log_var = tf.keras.layers.Dense(latent_dim)

        def call(self, inputs):
            x = self.dense_proj(inputs)
            mean = self.dense_mean(x)
            log_var = self.dense_log_var(x)
            return mean, log_var

    class Decoder(tf.keras.layers.Layer):
        def __init__(self, original_dim, hidden_dim, name="decoder", **kwargs):
            super().__init__(name=name, **kwargs)
            self.dense_proj = tf.keras.layers.Dense(hidden_dim, activation="relu")
            self.dense_output = tf.keras.layers.Dense(original_dim, activation="sigmoid")

        def call(self, inputs):
            x = self.dense_proj(inputs)
            return self.dense_output(x)

    class VAE(tf.keras.Model):
        def __init__(self, original_dim, latent_dim, hidden_dim, name="vae", **kwargs):
            super().__init__(name=name, **kwargs)
            self.encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
            self.decoder = Decoder(original_dim=original_dim, hidden_dim=hidden_dim)
            self.original_dim = original_dim
            self.latent_dim = latent_dim
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

        def call(self, inputs):
            mean, log_var = self.encoder(inputs)
            # Reparameterization trick
            batch = tf.shape(mean)[0]
            dim = tf.shape(mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            z = mean + tf.exp(0.5 * log_var) * epsilon
            return self.decoder(z), mean, log_var

        def train_step(self, data):
            with tf.GradientTape() as tape:
                reconstructed, mean, log_var = self(data)
                
                # Reconstruction loss (MSE)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstructed), axis=1)
                ) # Using binary_crossentropy for [0,1] scaled data

                # KL Divergence loss
                kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                
                total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "total_loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

    def build_vae(input_dim, latent_dim, hidden_dim_encoder=64, hidden_dim_decoder=64):
        """
        Builds and returns the VAE, Encoder, and Decoder models.

        Args:
            input_dim (int): Dimensionality of the input data.
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim_encoder (int): Number of units in the encoder's hidden layer.
            hidden_dim_decoder (int): Number of units in the decoder's hidden layer.

        Returns:
            tuple: (VAE model, Encoder model, Decoder model).
        """
        vae_model = VAE(original_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim_encoder)
        return vae_model, vae_model.encoder, vae_model.decoder
    ```
*   **Code Cell (Execution):**
    ```python
    # Define model parameters
    input_dim = scaled_financial_data.shape[1]
    latent_dim = 2 # A small latent dimension for easier visualization later
    hidden_dim = 64

    # Build the VAE model
    vae, encoder, decoder = build_vae(input_dim, latent_dim, hidden_dim)

    # Build the VAE by calling it once to create its weights
    dummy_input = tf.random.normal(shape=(1, input_dim))
    vae(dummy_input)

    print("VAE Model Summary:")
    vae.summary()
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The VAE architecture has been defined using Keras subclassing. We chose a `latent_dim` of 2 for simplicity, which allows us to potentially visualize the latent space later. The `Encoder` effectively compresses our financial features into a mean and log-variance vector, defining a Gaussian distribution in the latent space. The `Decoder` then expands a sample from this latent space back into the original feature dimension. The `sigmoid` activation on the decoder's output ensures that the generated data remains within the $ [0, 1] $ range, mirroring our scaled input. The `VAE` class itself incorporates the reparameterization trick and defines the custom `train_step` to handle both reconstruction and KL divergence losses.
    ```

### Section 6: Training the VAE Model

*   **Markdown Cell:**
    ```markdown
    Training the VAE involves minimizing the combined reconstruction and KL divergence loss. We will use the `Adam` optimizer, a popular choice for deep learning models due to its efficiency and adaptive learning rate capabilities. The training process will iterate over the dataset for a specified number of epochs, with data batched for computational efficiency. The `train_step` method within our `VAE` class handles the forward pass, loss calculation, and backpropagation for each batch.
    ```
*   **Code Cell (Function):**
    ```python
    def train_vae_model(vae_model, train_data, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Compiles and trains the VAE model.

        Args:
            vae_model (tf.keras.Model): The VAE model instance.
            train_data (np.array): The training data (scaled).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the Adam optimizer.

        Returns:
            tf.keras.callbacks.History: The training history object.
        """
        vae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        history = vae_model.fit(train_data, epochs=epochs, batch_size=batch_size, verbose=0)
        return history
    ```
*   **Code Cell (Execution):**
    ```python
    # Train the VAE model
    epochs = 100
    batch_size = 32
    learning_rate = 0.001

    print(f"Training VAE for {epochs} epochs with batch size {batch_size} and learning rate {learning_rate}...")
    vae_history = train_vae_model(vae, scaled_financial_data.values, epochs, batch_size, learning_rate)
    print("VAE training complete.")
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The VAE model has been trained on our scaled financial data. During training, the model iteratively adjusted its parameters to both accurately reconstruct the input data and ensure its latent space follows a standard normal distribution. The `vae_history` object contains the loss values recorded at each epoch, which we will visualize next to understand the training progression. These loss metrics are crucial indicators of how well the VAE is learning to capture the underlying data distribution.
    ```

### Section 7: Visualizing VAE Training Progress

*   **Markdown Cell:**
    ```markdown
    Visualizing the training losses is essential for monitoring the learning process of the VAE. We expect the reconstruction loss to decrease, indicating the decoder is getting better at reproducing the input. Simultaneously, the KL divergence loss should also decrease or stabilize, showing that the encoder is learning a latent distribution that closely approximates the prior. The total VAE loss, being the sum, should generally trend downwards, indicating overall model improvement.
    ```
*   **Code Cell (Function):**
    ```python
    def plot_vae_loss(history):
        """
        Plots the VAE reconstruction loss, KL divergence loss, and total VAE loss over epochs.

        Args:
            history (tf.keras.callbacks.History): The history object returned from VAE training.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
        plt.plot(history.history['kl_loss'], label='KL Divergence Loss')
        plt.plot(history.history['total_loss'], label='Total VAE Loss')
        plt.title('VAE Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    ```
*   **Code Cell (Execution):**
    ```python
    # Plot the VAE training losses
    plot_vae_loss(vae_history)
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The plot shows the progression of the VAE's reconstruction, KL divergence, and total losses over epochs. A consistent decrease in all loss components indicates that the VAE is learning effectively: the reconstruction loss is minimizing the error between original and reconstructed data, while the KL divergence is ensuring a well-structured and easily samplable latent space. This confirms that our VAE has learned a meaningful representation of the financial data and is ready to generate synthetic samples.
    ```

### Section 8: Generating Synthetic Data with VAE

*   **Markdown Cell:**
    ```markdown
    With a trained VAE, we can now generate new synthetic financial data. This process leverages the generative power of the VAE's decoder. By sampling random points from the standard normal distribution (which our encoder was encouraged to approximate for the latent space prior) and feeding these samples into the decoder, we can create novel data points that share the statistical characteristics of our original dataset. Importantly, we then apply the inverse transformation using the `MinMaxScaler` to bring the synthetic data back to its original interpretable scale.
    ```
*   **Code Cell (Function):**
    ```python
    def generate_vae_synthetic_data(decoder_model, num_samples, latent_dim, scaler, feature_names):
        """
        Generates synthetic data using the VAE decoder.

        Args:
            decoder_model (tf.keras.Model): The trained VAE decoder model.
            num_samples (int): The number of synthetic samples to generate.
            latent_dim (int): The dimensionality of the latent space.
            scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for inverse transformation.
            feature_names (list): List of original feature names.

        Returns:
            pandas.DataFrame: A DataFrame containing the inverse-transformed synthetic data.
        """
        # Generate random samples from a standard normal distribution for the latent vectors
        random_latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))
        
        # Pass these latent vectors through the decoder model
        generated_scaled_data = decoder_model.predict(random_latent_vectors, verbose=0)
        
        # Inverse transform the generated data to the original scale
        synthetic_data_array = scaler.inverse_transform(generated_scaled_data)
        
        return pd.DataFrame(synthetic_data_array, columns=feature_names)
    ```
*   **Code Cell (Execution):**
    ```python
    # Generate 1000 synthetic samples using the VAE
    vae_synthetic_data = generate_vae_synthetic_data(decoder, num_samples, latent_dim, scaler, financial_data.columns)

    print("First 5 rows of the VAE synthetic data:")
    print(vae_synthetic_data.head())

    print("\nDescriptive statistics of the VAE synthetic data:")
    print(vae_synthetic_data.describe())
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    We have successfully generated a new set of synthetic financial data using our trained VAE. Observing the head and descriptive statistics, the generated data now resides in the original financial scale, making it directly comparable to our real data. The values appear plausible and within reasonable financial ranges, indicating that the VAE has learned to capture the essence of the financial features and their underlying distribution.
    ```

### Section 9: Introduction to Generative Adversarial Networks (GANs)

*   **Markdown Cell:**
    ```markdown
    **Generative Adversarial Networks (GANs)**, introduced by Goodfellow et al. (2014), offer an alternative and highly effective approach to synthetic data generation. GANs are based on a **game-theoretic framework** involving two competing neural networks:

    1.  **Generator (G):** This network takes random noise $ \mathbf{z} $ as input and transforms it into synthetic data $ G(\mathbf{z}) $. Its goal is to produce data that is indistinguishable from real data.
    2.  **Discriminator (D):** This network takes both real data $ \mathbf{x} $ and synthetic data $ G(\mathbf{z}) $ as input and outputs a probability indicating whether the input is real (close to 1) or fake (close to 0). Its goal is to accurately distinguish between real and fake samples.

    The two networks are trained simultaneously in a **zero-sum game**: the Generator tries to maximize the Discriminator's error (i.e., fool it into thinking synthetic data is real), while the Discriminator tries to minimize its own error (i.e., correctly classify real and fake data). This adversarial process drives both networks to improve. Training is considered successful when the Generator can produce data that the Discriminator classifies as real with a probability of approximately $ 0.5 $, meaning it can no longer reliably distinguish between real and synthetic data.

    The objective function for a GAN is given by:
    $$ \min_G \max_D V(D, G) = E_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log D(\mathbf{x})] + E_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))] $$
    *   The Discriminator aims to maximize $ V(D, G) $, correctly identifying real samples as real ($ D(\mathbf{x}) \rightarrow 1 $) and fake samples as fake ($ D(G(\mathbf{z})) \rightarrow 0 $).
    *   The Generator aims to minimize $ V(D, G) $ (or equivalently, maximize $ E_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}[\log D(G(\mathbf{z}))] $), fooling the Discriminator into classifying fake samples as real ($ D(G(\mathbf{z})) \rightarrow 1 $).

    This adversarial dynamic allows GANs to learn complex, high-dimensional data distributions effectively, often generating highly realistic synthetic samples.
    ```

### Section 10: Defining the GAN Architecture

*   **Markdown Cell:**
    ```markdown
    Similar to the VAE, we'll define the Generator and Discriminator networks for our GAN using TensorFlow Keras.

    The **Generator** will take a random latent vector (noise) and upscale it through several `tf.keras.layers.Dense` layers with `relu` activations to generate data matching the dimensions of our financial features. Its final layer will use a `sigmoid` activation to output values within the $ [0, 1] $ range.

    The **Discriminator** will take either real or synthetic financial data as input and process it through `tf.keras.layers.Dense` layers with `relu` activations. Its final layer will be a single `Dense` unit with a `sigmoid` activation, outputting a probability score indicating the likelihood of the input being real.
    ```
*   **Code Cell (Function):**
    ```python
    def build_generator(latent_dim, output_dim, hidden_dim=64):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(latent_dim,)),
            tf.keras.layers.Dense(hidden_dim * 2, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='sigmoid') # Output scaled data [0,1]
        ], name='generator')
        return model

    def build_discriminator(input_dim, hidden_dim=64):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim * 2, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid') # Output probability (real/fake)
        ], name='discriminator')
        return model

    def build_gan(input_dim, latent_dim, hidden_dim_generator=64, hidden_dim_discriminator=64):
        """
        Builds and returns the GAN Generator, Discriminator, and the combined GAN model.

        Args:
            input_dim (int): Dimensionality of the real data.
            latent_dim (int): Dimensionality of the latent noise vector.
            hidden_dim_generator (int): Number of units in generator's hidden layer.
            hidden_dim_discriminator (int): Number of units in discriminator's hidden layer.

        Returns:
            tuple: (Generator model, Discriminator model, combined GAN model for generator training).
        """
        generator = build_generator(latent_dim, input_dim, hidden_dim_generator)
        discriminator = build_discriminator(input_dim, hidden_dim_discriminator)

        # The combined GAN model (for training the Generator)
        discriminator.trainable = False # Discriminator is frozen when training generator
        gan_input = tf.keras.Input(shape=(latent_dim,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan_model = tf.keras.Model(gan_input, gan_output, name='gan_model')

        return generator, discriminator, gan_model
    ```
*   **Code Cell (Execution):**
    ```python
    # Define model parameters
    input_dim = scaled_financial_data.shape[1]
    latent_dim = 20 # A larger latent dimension for GANs typically helps with expressiveness
    hidden_dim = 64

    # Build the GAN models
    generator, discriminator, gan_model = build_gan(input_dim, latent_dim, hidden_dim)

    print("Generator Model Summary:")
    generator.summary()

    print("\nDiscriminator Model Summary:")
    discriminator.summary()

    print("\nCombined GAN Model Summary (for generator training):")
    gan_model.summary()
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The Generator and Discriminator architectures for our GAN have been defined. The Generator is designed to transform random noise into synthetic financial features, while the Discriminator is set up to classify inputs as either real or synthetic. Note the `latent_dim` for GANs is typically chosen to be larger than VAEs to provide more expressive power to the generator. The combined GAN model facilitates the adversarial training process, ensuring that the Discriminator's weights are frozen when the Generator is being optimized to only update the Generator's parameters based on the Discriminator's feedback.
    ```

### Section 11: Training the GAN Model

*   **Markdown Cell:**
    ```markdown
    Training a GAN is an iterative and delicate process, involving alternating updates for the Discriminator and the Generator. This **adversarial training loop** is crucial:
    1.  **Discriminator Training:** The Discriminator is trained to distinguish between real data (labeled as `1`) and fake data generated by the current Generator (labeled as `0`).
    2.  **Generator Training:** The Generator is then trained to fool the Discriminator. It generates fake data, and the Discriminator's output for this fake data is used as a loss signal. The Generator's objective is to make the Discriminator output `1` (real) for its synthetic samples.

    This ensures that both networks continuously improve. We will use the `Adam` optimizer for both the Generator and Discriminator, and `tf.keras.losses.BinaryCrossentropy` as the loss function, as this is a binary classification task for the Discriminator.
    ```
*   **Code Cell (Function):**
    ```python
    def train_gan_model(generator, discriminator, gan_model, train_data, epochs=50, batch_size=32, latent_dim=100, learning_rate=0.0002):
        """
        Implements the adversarial training loop for the GAN.

        Args:
            generator (tf.keras.Model): The GAN generator model.
            discriminator (tf.keras.Model): The GAN discriminator model.
            gan_model (tf.keras.Model): The combined GAN model (generator + discriminator, with discriminator frozen).
            train_data (np.array): The training data (scaled).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            latent_dim (int): Dimensionality of the latent noise vector.
            learning_rate (float): Learning rate for the Adam optimizer.

        Returns:
            dict: A dictionary containing lists of discriminator and generator losses per epoch.
        """
        # Define optimizers and loss for discriminator and generator
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        discriminator.compile(optimizer=d_optimizer, loss=loss_fn)
        # Re-compile gan_model to ensure discriminator is non-trainable during generator training
        discriminator.trainable = False
        gan_model.compile(optimizer=g_optimizer, loss=loss_fn)
        discriminator.trainable = True # Re-enable for its own training steps

        d_losses = []
        g_losses = []

        dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(num_samples).batch(batch_size)

        for epoch in range(epochs):
            d_epoch_loss = tf.keras.metrics.Mean()
            g_epoch_loss = tf.keras.metrics.Mean()

            for batch in dataset:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Generate fake images
                noise = tf.random.normal(shape=(batch_size, latent_dim))
                generated_data = generator(noise)

                # Combine real and fake images
                combined_data = tf.concat([batch, generated_data], axis=0)
                
                # Labels for real and fake data
                labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
                
                # Add random noise to the labels to help the discriminator learn better
                labels += 0.05 * tf.random.uniform(tf.shape(labels))

                with tf.GradientTape() as tape:
                    predictions = discriminator(combined_data)
                    d_loss = loss_fn(labels, predictions)
                
                grads = tape.gradient(d_loss, discriminator.trainable_weights)
                d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))
                d_epoch_loss.update_state(d_loss)

                # ---------------------
                #  Train Generator
                # ---------------------
                # Generate random noise
                noise = tf.random.normal(shape=(batch_size, latent_dim))
                
                # Labels for generator training (want discriminator to classify as real)
                misleading_labels = tf.ones((batch_size, 1))

                # Train the generator (discriminator weights are frozen)
                with tf.GradientTape() as tape:
                    generated_data = generator(noise)
                    predictions = discriminator(generated_data)
                    g_loss = loss_fn(misleading_labels, predictions)
                
                grads = tape.gradient(g_loss, generator.trainable_weights)
                g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
                g_epoch_loss.update_state(g_loss)
            
            d_losses.append(d_epoch_loss.result().numpy())
            g_losses.append(g_epoch_loss.result().numpy())
            # print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_epoch_loss.result():.4f}, G Loss: {g_epoch_loss.result():.4f}")
            
        return {'d_loss': d_losses, 'g_loss': g_losses}
    ```
*   **Code Cell (Execution):**
    ```python
    # Train the GAN model
    epochs = 100
    batch_size = 32
    latent_dim_gan = 20 # Use the latent_dim defined for GAN
    learning_rate = 0.0002

    print(f"Training GAN for {epochs} epochs with batch size {batch_size} and latent dimension {latent_dim_gan}...")
    gan_history = train_gan_model(generator, discriminator, gan_model, scaled_financial_data.values, epochs, batch_size, latent_dim_gan, learning_rate)
    print("GAN training complete.")
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The GAN model has undergone its adversarial training process. This iterative competition between the Generator and Discriminator is what enables GANs to learn to produce highly realistic data. The `gan_history` dictionary stores the average losses for both networks across each epoch, which are crucial for assessing how well the adversarial training progressed. Observing these losses will give us insight into the stability and effectiveness of the training. A balanced training (where neither loss dominates) is often indicative of good convergence.
    ```

### Section 12: Visualizing GAN Training Progress

*   **Markdown Cell:**
    ```markdown
    Monitoring the Discriminator and Generator losses during GAN training helps us understand the dynamics of the adversarial game. While VAE losses typically show a smooth downward trend, GAN losses can be more volatile due to the competing nature of the networks. Ideally, we want to see both losses converge to a stable state, indicating neither network is overwhelmingly winning, and the Generator is producing data that the Discriminator struggles to classify. If one loss consistently goes to zero while the other remains high, it might indicate issues like mode collapse or training instability.
    ```
*   **Code Cell (Function):**
    ```python
    def plot_gan_loss(history):
        """
        Plots the Generator and Discriminator loss curves over epochs.

        Args:
            history (dict): A dictionary containing 'd_loss' and 'g_loss' lists.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['d_loss'], label='Discriminator Loss')
        plt.plot(history['g_loss'], label='Generator Loss')
        plt.title('GAN Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    ```
*   **Code Cell (Execution):**
    ```python
    # Plot the GAN training losses
    plot_gan_loss(gan_history)
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The plot illustrates the Generator and Discriminator losses over the training epochs. Fluctuations are common in GAN training, but a general trend where both losses reach a relatively stable, non-zero point suggests that the Generator is producing data realistic enough to challenge the Discriminator. If one loss consistently goes to zero while the other remains high, it might indicate a mode collapse (Generator has found a few samples that fool the discriminator but doesn't capture full data diversity) or training instability where one network dominates the other. For our current plot, we observe a competitive balance, indicating effective adversarial learning.
    ```

### Section 13: Generating Synthetic Data with GAN

*   **Markdown Cell:**
    ```markdown
    Once the GAN is trained, the Generator network alone is used to produce synthetic data. We simply feed random noise vectors into the trained Generator, and it will output new data points that statistically resemble the real financial data it learned from. Similar to the VAE, the generated data is then inverse-transformed using our `MinMaxScaler` to return it to the original, interpretable financial scale.
    ```
*   **Code Cell (Function):**
    ```python
    def generate_gan_synthetic_data(generator_model, num_samples, latent_dim, scaler, feature_names):
        """
        Generates synthetic data using the GAN generator.

        Args:
            generator_model (tf.keras.Model): The trained GAN generator model.
            num_samples (int): The number of synthetic samples to generate.
            latent_dim (int): The dimensionality of the latent noise vector.
            scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for inverse transformation.
            feature_names (list): List of original feature names.

        Returns:
            pandas.DataFrame: A DataFrame containing the inverse-transformed synthetic data.
        """
        # Generate random samples from a standard normal distribution for the input noise
        random_noise = tf.random.normal(shape=(num_samples, latent_dim))
        
        # Pass these noise vectors through the generator model
        generated_scaled_data = generator_model.predict(random_noise, verbose=0)
        
        # Inverse transform the generated data to the original scale
        synthetic_data_array = scaler.inverse_transform(generated_scaled_data)
        
        return pd.DataFrame(synthetic_data_array, columns=feature_names)
    ```
*   **Code Cell (Execution):**
    ```python
    # Generate 1000 synthetic samples using the GAN
    gan_synthetic_data = generate_gan_synthetic_data(generator, num_samples, latent_dim_gan, scaler, financial_data.columns)

    print("First 5 rows of the GAN synthetic data:")
    print(gan_synthetic_data.head())

    print("\nDescriptive statistics of the GAN synthetic data:")
    print(gan_synthetic_data.describe())
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    We have now produced synthetic financial data using our trained GAN Generator. The initial rows and descriptive statistics show that the data is scaled back to its original range, allowing for direct comparison. These samples represent novel financial scenarios that maintain the patterns and distributions learned from the real dataset. With both VAE and GAN synthetic datasets ready, we can proceed to a detailed comparative statistical analysis.
    ```

### Section 14: Comparative Statistical Analysis of Real vs. Synthetic Data

*   **Markdown Cell:**
    ```markdown
    A crucial step in validating synthetic data is to compare its statistical properties against the original real data. This helps us quantify how well our VAE and GAN models have captured the underlying data distribution and relationships. We will focus on:
    *   **Descriptive Statistics**: Comparing means, standard deviations, and value ranges for each feature.
    *   **Distribution Shapes**: Visualizing overlaid histograms and Kernel Density Estimates (KDEs) for each feature to see how well the distributions match.
    *   **Visual Patterns**: For features that might exhibit time-series-like behavior (even if generated samples are independent), a visual inspection can provide qualitative insights into their structure.
    ```
*   **Code Cell (Function):**
    ```python
    def compare_statistics(real_data, synthetic_data_vae, synthetic_data_gan):
        """
        Compares descriptive statistics of real, VAE synthetic, and GAN synthetic data.

        Args:
            real_data (pandas.DataFrame): The original real financial data.
            synthetic_data_vae (pandas.DataFrame): Synthetic data generated by VAE.
            synthetic_data_gan (pandas.DataFrame): Synthetic data generated by GAN.
        """
        print("--- Descriptive Statistics Comparison ---")
        
        print("\nReal Data Statistics:")
        print(real_data.describe())
        
        print("\nVAE Synthetic Data Statistics:")
        print(synthetic_data_vae.describe())
        
        print("\nGAN Synthetic Data Statistics:")
        print(synthetic_data_gan.describe())

    def compare_distributions(real_data, synthetic_data_vae, synthetic_data_gan, feature_names):
        """
        Compares feature distributions using overlaid histograms and KDE plots.

        Args:
            real_data (pandas.DataFrame): The original real financial data.
            synthetic_data_vae (pandas.DataFrame): Synthetic data generated by VAE.
            synthetic_data_gan (pandas.DataFrame): Synthetic data generated by GAN.
            feature_names (list): List of feature names to plot.
        """
        print("\n--- Feature Distribution Comparison (Histograms & KDEs) ---")
        num_features = len(feature_names)
        num_cols = 2
        num_rows = (num_features + num_cols - 1) // num_cols

        plt.figure(figsize=(num_cols * 7, num_rows * 5))
        for i, feature in enumerate(feature_names):
            plt.subplot(num_rows, num_cols, i + 1)
            sns.histplot(real_data[feature], color='blue', label='Real', kde=True, stat='density', alpha=0.5, line_kws={'linewidth':2})
            sns.histplot(synthetic_data_vae[feature], color='green', label='VAE Synthetic', kde=True, stat='density', alpha=0.5, line_kws={'linewidth':2})
            sns.histplot(synthetic_data_gan[feature], color='red', label='GAN Synthetic', kde=True, stat='density', alpha=0.5, line_kws={'linewidth':2})
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_time_series_comparison(real_data, synthetic_data_vae, synthetic_data_gan, feature_name, num_samples_to_plot=100):
        """
        Plots time-series like comparisons for a selected feature.

        Args:
            real_data (pandas.DataFrame): The original real financial data.
            synthetic_data_vae (pandas.DataFrame): Synthetic data generated by VAE.
            synthetic_data_gan (pandas.DataFrame): Synthetic data generated by GAN.
            feature_name (str): The name of the feature to plot.
            num_samples_to_plot (int): Number of samples to plot for visual comparison.
        """
        print(f"\n--- Time-Series Like Comparison for '{feature_name}' ---")
        plt.figure(figsize=(15, 6))
        
        plt.plot(real_data[feature_name].head(num_samples_to_plot), label='Real Data', color='blue', alpha=0.7)
        plt.plot(synthetic_data_vae[feature_name].head(num_samples_to_plot), label='VAE Synthetic', color='green', linestyle='--', alpha=0.7)
        plt.plot(synthetic_data_gan[feature_name].head(num_samples_to_plot), label='GAN Synthetic', color='red', linestyle=':', alpha=0.7)
        
        plt.title(f'Time-Series Like Visual Comparison of {feature_name}')
        plt.xlabel('Sample Index')
        plt.ylabel(feature_name)
        plt.legend()
        plt.grid(True)
        plt.show()
    ```
*   **Code Cell (Execution):**
    ```python
    # Compare descriptive statistics
    compare_statistics(financial_data, vae_synthetic_data, gan_synthetic_data)

    # Compare feature distributions
    feature_names = financial_data.columns.tolist()
    compare_distributions(financial_data, vae_synthetic_data, gan_synthetic_data, feature_names)

    # Plot time-series like comparison for 'Daily_Return'
    plot_time_series_comparison(financial_data, vae_synthetic_data, gan_synthetic_data, 'Daily_Return')

    # Plot time-series like comparison for 'Volatility'
    plot_time_series_comparison(financial_data, vae_synthetic_data, gan_synthetic_data, 'Volatility')
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    The statistical comparisons provide critical insights into the quality of the generated synthetic data.
    *   From the **descriptive statistics tables**, Financial Data Engineers can quantitatively assess how closely the means, standard deviations, and value ranges of synthetic data align with the real data. Deviations here can indicate a lack of fidelity.
    *   The **distribution plots** offer a visual assessment, showing whether the models have captured the overall shape and spread of each feature's distribution. Ideally, the synthetic distributions should closely overlap with the real data's distribution. This is a strong indicator of how well the models have learned the underlying data generating process.
    *   The **time-series like plots** provide a qualitative view. While our basic VAE/GAN models don't explicitly model temporal dependencies, these plots allow us to visually inspect if the *characteristics* of the sequential patterns (e.g., typical volatility levels, return fluctuations, or the smoothness/choppiness of a feature) are maintained. This helps in understanding if the generated data *looks* like plausible financial time series, even without perfect autocorrelation.

    By analyzing these comparisons, Financial Data Engineers can determine which model (VAE or GAN) is more effective for their specific data and application, based on how accurately it reproduces the essential statistical signatures of the real financial environment.
    ```

### Section 15: Adjusting Latent Variables / Noise for Data Variation

*   **Markdown Cell:**
    ```markdown
    One of the powerful aspects of generative models is the ability to influence the characteristics of the generated data by manipulating their input: the latent variables for VAEs or the noise vector for GANs. This interactivity allows Financial Data Engineers to explore various hypothetical market scenarios or generate data with specific properties, which is invaluable for stress-testing and scenario analysis.

    For **VAEs**, we can sample from specific points in the learned latent space, or slightly perturb a central latent vector (e.g., the mean of the encoded real data's latent space) to generate data variations around a 'typical' financial state.

    For **GANs**, we can vary the distribution or scale of the input noise vector $ \mathbf{z} $. For example, we might sample from a different standard deviation or a skewed distribution to simulate more extreme or unusual market conditions. This provides a mechanism for **scenario generation**.
    ```
*   **Code Cell (Function):**
    ```python
    def explore_vae_latent_space(decoder_model, latent_dim, scaler, original_data_stats, feature_names, num_variations=5, latent_dim_to_vary=0):
        """
        Generates and analyzes synthetic data by perturbing VAE latent vectors.

        Args:
            decoder_model (tf.keras.Model): The trained VAE decoder model.
            latent_dim (int): Dimensionality of the latent space.
            scaler (sklearn.preprocessing.MinMaxScaler): The scaler for inverse transformation.
            original_data_stats (dict): Dictionary of mean statistics for original data features.
            feature_names (list): List of original feature names.
            num_variations (int): Number of variations to generate.
            latent_dim_to_vary (int): Index of the latent dimension to perturb.
        """
        print(f"--- Exploring VAE Latent Space by varying latent dimension {latent_dim_to_vary} ---")
        
        # Start with a central latent vector (e.g., zeros)
        base_latent_vector = np.zeros((1, latent_dim))
        
        variation_results = []
        variation_labels = []
        
        # Define perturbation range
        perturbations = np.linspace(-3, 3, num_variations) # Varies across +/- 3 standard deviations in latent space
        
        for i, p in enumerate(perturbations):
            current_latent_vector = base_latent_vector.copy()
            current_latent_vector[0, latent_dim_to_vary] = p
            
            generated_scaled_data = decoder_model.predict(current_latent_vector, verbose=0)
            synthetic_sample = scaler.inverse_transform(generated_scaled_data)[0]
            synthetic_df = pd.DataFrame([synthetic_sample], columns=feature_names)
            
            variation_results.append(synthetic_df)
            variation_labels.append(f'Perturbation {p:.2f}')
            
            print(f"\nVariation {i+1} (Latent[{latent_dim_to_vary}] = {p:.2f}):")
            print(synthetic_df.round(4))

        # Optional: Plotting how a specific feature changes
        feature_to_plot = 'Daily_Return' # Example feature
        if feature_to_plot in feature_names:
            plt.figure(figsize=(10, 5))
            feature_values = [res[feature_to_plot].iloc[0] for res in variation_results]
            plt.plot(perturbations, feature_values, marker='o')
            plt.axhline(y=original_data_stats[feature_to_plot], color='r', linestyle='--', label=f'Original Mean {feature_to_plot}')
            plt.title(f'Effect of Latent Dimension {latent_dim_to_vary} Variation on {feature_to_plot}')
            plt.xlabel(f'Latent Dimension {latent_dim_to_vary} Value')
            plt.ylabel(feature_to_plot)
            plt.legend()
            plt.grid(True)
            plt.show()


    def explore_gan_noise_variation(generator_model, latent_dim, scaler, original_data_stats, feature_names, num_variations=5, noise_factor_range=(0.5, 2.0)):
        """
        Generates and analyzes synthetic data by varying GAN input noise.

        Args:
            generator_model (tf.keras.Model): The trained GAN generator model.
            latent_dim (int): Dimensionality of the latent noise vector.
            scaler (sklearn.preprocessing.MinMaxScaler): The scaler for inverse transformation.
            original_data_stats (dict): Dictionary of mean statistics for original data features.
            feature_names (list): List of original feature names.
            num_variations (int): Number of variations to generate.
            noise_factor_range (tuple): (min_factor, max_factor) to scale random noise.
        """
        print(f"--- Exploring GAN Noise Variation (scaling noise by factor) ---")

        # Generate a base random noise vector
        base_noise = tf.random.normal(shape=(1, latent_dim))

        variation_results = []
        variation_labels = []
        
        # Define noise scaling factors
        noise_factors = np.linspace(noise_factor_range[0], noise_factor_range[1], num_variations)
        
        for i, factor in enumerate(noise_factors):
            current_noise = base_noise * factor
            
            generated_scaled_data = generator_model.predict(current_noise, verbose=0)
            synthetic_sample = scaler.inverse_transform(generated_scaled_data)[0]
            synthetic_df = pd.DataFrame([synthetic_sample], columns=feature_names)
            
            variation_results.append(synthetic_df)
            variation_labels.append(f'Noise Factor {factor:.2f}')
            
            print(f"\nVariation {i+1} (Noise Factor = {factor:.2f}):")
            print(synthetic_df.round(4))

        # Optional: Plotting how a specific feature changes
        feature_to_plot = 'Volatility' # Example feature
        if feature_to_plot in feature_names:
            plt.figure(figsize=(10, 5))
            feature_values = [res[feature_to_plot].iloc[0] for res in variation_results]
            plt.plot(noise_factors, feature_values, marker='o')
            plt.axhline(y=original_data_stats[feature_to_plot], color='r', linestyle='--', label=f'Original Mean {feature_to_plot}')
            plt.title(f'Effect of Noise Factor Variation on {feature_to_plot}')
            plt.xlabel('Noise Scaling Factor')
            plt.ylabel(feature_to_plot)
            plt.legend()
            plt.grid(True)
            plt.show()
    ```
*   **Code Cell (Execution):**
    ```python
    # Get original data statistics (mean of each feature) for comparison
    original_stats_mean = financial_data.mean().to_dict()
    
    # Explore VAE latent space variations
    explore_vae_latent_space(decoder, latent_dim, scaler, original_stats_mean, financial_data.columns, num_variations=7, latent_dim_to_vary=0)

    # Explore GAN noise variations
    explore_gan_noise_variation(generator, latent_dim_gan, scaler, original_stats_mean, financial_data.columns, num_variations=7, noise_factor_range=(0.2, 3.0))
    ```
*   **Markdown Cell (Explanation):**
    ```markdown
    By adjusting the latent variables for the VAE and the input noise for the GAN, we can observe how these changes propagate to the generated synthetic financial data. This demonstrates the "interactive" aspect of these generative models.
    *   For the VAE, a small shift in a specific latent dimension can smoothly change the generated `Daily_Return` or `Volatility`, allowing us to simulate slightly different market conditions.
    *   For the GAN, scaling the input noise vector by different factors can generate data with varying levels of extremity. For instance, increasing the "noise factor" might produce more volatile or extreme synthetic data points, simulating tail events or heightened market uncertainty.

    This capability is vital for Financial Data Engineers conducting scenario analysis and stress-testing, allowing them to probe the models for various hypothetical outcomes and understand the resilience of financial models under diverse, user-defined conditions.
    ```

### Section 16: Financial Applications of Synthetic Data

*   **Markdown Cell:**
    ```markdown
    The ability to generate high-fidelity synthetic financial data using VAEs and GANs opens up a multitude of critical applications for Financial Data Engineers:

    1.  **Privacy-Preserving Data Sharing:** Real financial data often contains highly sensitive information (e.g., client portfolios, proprietary trading strategies) subject to strict regulations (GDPR, CCPA). Synthetic data allows for the creation of anonymized, non-identifiable datasets that maintain the statistical properties of the original data. This enables secure sharing with third parties, academic researchers, or across departments for model development and testing without compromising privacy or regulatory compliance.

    2.  **Stress-Testing Financial Models:** Financial models (e.g., risk models, pricing models, trading algorithms) need to be robust to extreme and unprecedented market events. Historical data, by definition, lacks records of future crises. Synthetic data generators can be specifically trained or manipulated (as demonstrated in the previous section by adjusting latent variables/noise) to produce data reflecting "black swan" events, severe economic downturns, or sudden market shocks. This allows Financial Data Engineers to stress-test models beyond historical limits and assess their resilience in hypothetical adverse scenarios.

    3.  **Simulating Complex Market Scenarios:** Financial markets are dynamic and influenced by numerous interacting factors. Synthetic data can be used to simulate hypothetical market scenarios (e.g., persistent low interest rates, high inflation environments, sector-specific shocks, or varying market sentiment) to evaluate the performance of investment strategies, optimize portfolio allocations, or assess systemic risk under various conditions. This capability enables proactive decision-making and strategic planning.

    4.  **Enhancing Data Augmentation and Scarcity:** In many financial domains, high-quality labeled data can be scarce or expensive to acquire (e.g., rare fraud events, specific option types). Generative models can augment existing datasets by creating new, realistic samples, effectively increasing the training data size and improving the generalization capabilities of machine learning models for tasks like fraud detection, credit scoring, or algorithmic trading.

    5.  **Uncovering Latent Factors Driving Asset Prices:** By compressing high-dimensional financial data into a lower-dimensional latent space, VAEs can implicitly learn and reveal hidden, uncorrelated factors that drive asset prices or market behavior. These latent factors can provide novel insights for portfolio construction, risk management, and understanding market dynamics, going beyond traditional observable economic indicators.

    In conclusion, VAEs and GANs provide powerful tools for Financial Data Engineers to innovate, manage risk, and extract value from data in a privacy-preserving and robust manner, addressing some of the most pressing challenges in modern finance.
    ```

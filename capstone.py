import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as pltmport

# Load the dataset
# Assuming 'data.csv' contains ID, TITLE, ABSTRACT, and TOPIC columns
data = pd.read_csv('/content/a.csv')

# Preprocessing to tokenize the text
documents = data['ABSTRACT'].tolist()

# Convert documents to a matrix of token counts
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(documents).toarray()

# Encode topic labels
encoder = LabelEncoder()
y = encoder.fit_transform(data['TITLE'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define VAE architecture
original_dim = X_train.shape[1]
latent_dim = 10

# Encoder network
inputs = Input(shape=(original_dim,), name='encoder_input')
h1 = Dense(128, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
z_mean = Dense(latent_dim, name='z_mean')(h2)
z_log_var = Dense(latent_dim, name='z_log_var')(h2)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Sample latent points
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Decoder network
decoder_h1 = Dense(64, activation='relu')
decoder_h2 = Dense(128, activation='relu')
decoder_out = Dense(original_dim, activation='sigmoid')

h_decoded = decoder_h1(z)
h_decoded = decoder_h2(h_decoded)
x_decoded_mean = decoder_out(h_decoded)

# Define VAE model
vae = Model(inputs, x_decoded_mean)

# Define loss function
reconstruction_loss = mse(inputs, x_decoded_mean)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile VAE model
optimizer = Adam(lr=0.001)
vae.compile(optimizer=optimizer)

# Train VAE model and track loss
history = vae.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test))

# Extract loss from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
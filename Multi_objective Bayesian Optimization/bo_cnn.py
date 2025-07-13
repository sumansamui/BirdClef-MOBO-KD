import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import csv
from keras import optimizers
import keras
from functools import partial
# from keras.backend import sigmoid
from math import exp
from keras.utils import get_custom_objects
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report 
import joblib
import pickle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import cycle
##################### GPU Set up #####################
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set only the first GPU as visible
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Allow memory growth to allocate memory dynamically on the GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configuration successful.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")
from keras.mixed_precision import Policy
from keras.mixed_precision import set_global_policy

policy = Policy('mixed_float16')
set_global_policy(policy)

##################### Load the dataset #####################
mfcc_directory = './Birdclef/mel_spec'
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
def load_from_hdf5(filename, directory):
    """Load data from HDF5 file"""
    filepath = os.path.join(directory, filename)
    with h5py.File(filepath, 'r') as hf:
        data = {name: hf[name][:] for name in hf.keys()}


    if 'classes' in data:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = data['classes'].astype(str)
        data['label_encoder'] = label_encoder

    return data

loaded_data = load_from_hdf5('/home/22EC1102/soumen/data/kws_10_log_mel/bird_sound/mel_spec.h5', mfcc_directory)


X = loaded_data['X_train']
y = loaded_data['y_train']
label_encoder = loaded_data['label_encoder']
print(X.shape)
print(y.shape)


original_labels = label_encoder.inverse_transform(y)


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.5,  #0.1
    random_state=42,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.6,  #0.5
    random_state=42,
    stratify=y_temp
)
print(X_train.shape)
print(y_train.shape)
print(X_validation.shape)
print(y_validation.shape)
print(X_test.shape)
print(y_test.shape)

##################### Define a Simple KWS Model in TensorFlow #####################


from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

EPOCHS = 200
BATCH_SIZE = 16  
PATIENCE = 5  
LEARNING_RATE = 0.0001 
SKIP = 1
CLASS = 11
clipnorm=1.0
clipvalue=0.5
##################### Create a CNN model #####################
def create_2d_cnn_model(residual, filters, kernel_size, fc_layers, use_bn, use_dropout, input_shape):

    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Initial Conv Block
    x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="relu1_1")(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2), strides=(2,2), padding='same')(x)  # Changed to (2,2)


    # Residual Block 1
    for _ in range(residual - 1):
        filters = filters*2
        residual = layers.Conv2D(filters, (1,1), strides=(2,2), padding='same')(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2), strides=(2,2), padding='same')(x)  # Added pooling
        x = layers.add([x, residual])
        x = layers.ReLU()(x)

    # Final Feature Processing
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense Layers
    fc_layer_configs = {
        4: [1024, 512, 256, 128],  
        3: [512, 256, 128],  
        2: [256, 128],  
        1: [128]
        }
    
    num_fc_layers = fc_layers  # Example: 4, 3, 2, or 1

    if num_fc_layers in fc_layer_configs:
        for i, neurons in enumerate(fc_layer_configs[num_fc_layers]):  
            x = layers.Dense(neurons, activation='relu')(x)
            if use_dropout:
                x = layers.Dropout(0.2)(x)


    outputs = layers.Dense(CLASS, activation='softmax', name="output_layer")(x)

    # Compile
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        #clipnorm=clipnorm,
        clipvalue=clipvalue
    )

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    # model.summary()
    
    return model

##################### Train and Evaluate the Model #####################
def train_evaluate_model(model):

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=0)
    
    # Return validation accuracy (simulate it) and model size
    accuracy = history.history['val_accuracy'][-1]
    num_params = model.count_params()
    return accuracy, num_params

##################### Bayesian Optimization Setup Using Gaussian Processes (GPflow) #####################

import gpflow
from gpflow.utilities import print_summary
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import itertools
input_shape = X_train.shape[1:]


##################### Objective function to optimize (returns accuracy and negative params) #####################
def objective_function(residual, filters, kernel_size, fc_layers, use_bn, use_dropout):
    use_bn = bool(use_bn)
    use_dropout = bool(use_dropout)
    fc_layers = int(fc_layers)
    filters = int(filters)
    model = create_2d_cnn_model(residual, filters, kernel_size, fc_layers, use_bn, use_dropout, input_shape)

    acc, params = train_evaluate_model(model)
    # Return accuracy (maximize) and model size (minimize -> hence negative)
    return np.array([acc, -params])  # Return in structured format

##################### Define possible values #####################
residual = [ 1, 2]             
filters = [16, 32, 64]          
kernel_size = [(3, 3), (5, 5)]
fc_layers = [1, 2]              
use_bn = [1, 0]  	            # Representing True as 1 and False as 0
use_dropout = [1, 0]

##################### Generate all combinations of hyperparameters #####################
param_space = [
    {'residual': cl,'filters': f, 'kernel_size': ks, 'fc_layers': fc, 'use_bn': bn, 'use_dropout': do}
    for cl, f, ks, fc, bn, do in itertools.product(residual, filters, kernel_size, fc_layers, use_bn, use_dropout)
]


##################### Collect initial data #####################
X_init = np.array([[p['residual'],p['filters'], p['kernel_size'][0],p['kernel_size'][1], p['fc_layers'], p['use_bn'], p['use_dropout']] for p in param_space])
Y_init = np.array([objective_function(**p) for p in param_space])

##################### Verify data shapes #####################
print("X_init shape:", X_init.shape)
print("Y_init shape:", Y_init.shape)
print("First sample (X):", X_init[0])
print("First evaluation (Y):", Y_init[0])

##################### Gaussian Process Models #####################

##################### Scale the data #####################
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_init)

##################### Create Gaussian Process models for each objective #####################
kern_acc = gpflow.kernels.Matern52()
kern_size = gpflow.kernels.Matern52()

gp_acc = gpflow.models.GPR(data=(X_scaled, Y_init[:, 0:1]), kernel=kern_acc)
gp_size = gpflow.models.GPR(data=(X_scaled, Y_init[:, 1:2]), kernel=kern_size)

##################### Optimize GP hyperparameters #####################
gpflow.optimizers.Scipy().minimize(gp_acc.training_loss, gp_acc.trainable_variables)
gpflow.optimizers.Scipy().minimize(gp_size.training_loss, gp_size.trainable_variables)


##################### Bayesian Optimization Loop #####################
from scipy.optimize import minimize
import csv
import numpy as np
import pandas as pd

# Initialize a list to store hyperparameters, accuracy, and model size
pareto_data = []

#################################
# Function to save the collected Pareto front data to CSV
def save_pareto_data_to_csv(pareto_data, filename="pareto_front.csv"):
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(pareto_data)
    # Save the DataFrame to a CSV file
    df.to_csv(filename, mode='a', index=False)
    
##################################
##################### Acquisition function (e.g., Expected Improvement) #####################
def acquisition_function(x):
    # Reshape x to be 2D [1, D] before passing to GP models
    x_reshaped = x.reshape(1, -1)  # Shape should be [1, D]
    # Predict mean and variance for each GP model
    mu_acc, var_acc = gp_acc.predict_f(x_reshaped)
    mu_size, var_size = gp_size.predict_f(x_reshaped)
    # Calculate acquisition score based on objectives (simplified in this case)
    return - (mu_acc + mu_size)  # Simplified version for illustration

####################################

##################### Define your objective function, which evaluates the model using the given hyperparameters #####################
def objective_function_1(hyperparameters):
	residual = hyperparameters['residual']
	filters = hyperparameters['filters']
	kernel_size = hyperparameters['kernel_size']
	fc_layers = hyperparameters['fc_layers']
	use_bn = hyperparameters['use_bn']
	use_dropout = hyperparameters['use_dropout']
	# if isinstance(fc_layers, int):
	# 	fc_layers = [fc_layers]
     
	use_bn = bool(use_bn)
	use_dropout = bool(use_dropout)
	fc_layers = int(fc_layers)
	filters = int(filters)
     
	model = create_2d_cnn_model(residual, filters, kernel_size, fc_layers, use_bn, use_dropout, input_shape)
	accuracy, model_size = train_evaluate_model(model)  # Placeholder function
	return np.array([accuracy, model_size])  # Must match the format of Y_scaled

######################
##################### Function to map scaled values to the original hyperparameter values #####################

def map_to_hyperparameters(new_x, param_space):
    hyperparameters = {}

    ##################### Rescale the index of each parameter based on the scaled value #####################
    for i, param in enumerate(param_space[0].keys()):
        # Get the values for the current parameter across all configurations
        param_values = [x[param] for x in param_space]
        
        ##################### Clamp the value between 0 and 1 to avoid out-of-range errors #####################
        clamped_value = min(max(new_x[0, i], 0), 1)
        
        ##################### Map the clamped value to an index in the param_values list #####################
        idx = int(clamped_value * (len(param_values) - 1))
        
        ##################### Assign the parameter value to the hyperparameters dictionary #####################
        hyperparameters[param] = param_values[idx]

    return hyperparameters



##################### Bayesian optimization loop

n_iterations = 100    # Number of iterations for optimization
for i in range(n_iterations):
    ##################### Use a random point from X_scaled as the starting point #####################
    x0 = X_scaled[np.random.choice(X_scaled.shape[0]), :].flatten()  # Flatten to make it 1D array
    
    ##################### Minimize the acquisition function #####################
    result = minimize(acquisition_function, x0, method='L-BFGS-B')
    #print(result)
    
    ##################### Get the new point from the optimization result #####################
    new_x = result.x.reshape(1, -1)  # Ensure new_x is 2D
    #print(new_x)
    ##################### Map the optimized values to the real hyperparameter space #####################
    hyperparameters = map_to_hyperparameters(new_x, param_space)  # Use new_x directly, not result.x
    print(f"Iteration {i+1}: Hyperparameters = {hyperparameters}")

    ##################### Evaluate the new point using the objective function #####################
    new_y = objective_function_1(hyperparameters).reshape(1, -1)
    # print("new_x shape:", new_x.shape)
    # print("new_y shape:", new_y.shape)
    
    ##################### Update GPs with new data #####################
    X_scaled = np.vstack((X_scaled, new_x))
    Y_scaled = np.vstack((Y_init, new_y))    #  Y_scaled
    X_scaled = X_scaled[-len(Y_scaled):]    # Keep both arrays the same length
    # print("X_scaled shape:", X_scaled.shape)
    # print("Y_scaled shape (Accuracy):", Y_scaled[:, 0:1].shape)
    # print("Y_scaled shape (Model Size):", Y_scaled[:, 1:2].shape)
				    # Store the hyperparameters, accuracy, and model size
    pareto_data.append({
        "iterations": i,
        "residual": hyperparameters['residual'],
        "filters": hyperparameters['filters'],
        "kernel_size": hyperparameters['kernel_size'],
        "fc_layers": hyperparameters['fc_layers'],
        "use_bn": hyperparameters['use_bn'],
        "use_dropout": hyperparameters['use_dropout'],
        "accuracy": new_y[0, 0],  # Accuracy value
        "model_size": new_y[0, 1]  # Model size value
    })
    
    ##################### Recreate and re-optimize the GP models with updated data ##################### 
    gp_acc = gpflow.models.GPR(data=(X_scaled, Y_scaled[:, 0:1]), kernel=kern_acc)
    gp_size = gpflow.models.GPR(data=(X_scaled, Y_scaled[:, 1:2]), kernel=kern_size)
    gpflow.optimizers.Scipy().minimize(gp_acc.training_loss, gp_acc.trainable_variables)
    gpflow.optimizers.Scipy().minimize(gp_size.training_loss, gp_size.trainable_variables)

##################### Save the collected Pareto data to a CSV file after the loop finishes #####################
save_pareto_data_to_csv(pareto_data, "pareto_front.csv")

##################### Plot Pareto front #####################

plt.scatter(Y_scaled[:, 1], Y_scaled[:, 0], color='red')  # Note the negation of the model size for visualization
plt.xlabel('Model Size')
plt.ylabel('Accuracy')
plt.title('Pareto Front')
plt.show()

###################################
###################################
import pandas as pd

def is_dominated(candidate, front):
    """
    Check if the candidate solution is dominated by any solution in the Pareto front.
    candidate: The new candidate solution (accuracy, model_size).
    front: List of Pareto-optimal solutions so far.
    """
    for solution in front:
        # A solution 'solution' dominates 'candidate' if it is better in all objectives
        if all(c <= s for c, s in zip(candidate, solution)) and any(c < s for c, s in zip(candidate, solution)):
            return True  # candidate is dominated by solution
    return False  # candidate is not dominated

def get_pareto_front_from_csv(csv_file):
    """
    Extract the Pareto front from the CSV file containing all the solutions.
    csv_file: The path to the CSV file containing the solutions.
    """
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)
    
    # Extract accuracy and model size columns (assuming these columns exist)
    pareto_front = []
    for _, row in data.iterrows():
        candidate = [row['accuracy'], row['model_size']]
        
        # Update the Pareto front if the candidate is not dominated
        if not is_dominated(candidate, pareto_front):
            # Remove solutions dominated by the candidate
            pareto_front = [s for s in pareto_front if not is_dominated(s, [candidate])]
            # Add the candidate as a Pareto-optimal solution
            pareto_front.append(candidate)

    return pareto_front

# Path to the CSV file containing all the solutions
csv_file = "pareto_front.csv"  # Change this to your actual file path

# Get the Pareto front from the CSV data
pareto_front = get_pareto_front_from_csv(csv_file)

# Save the Pareto front to a new CSV file
pareto_front_df = pd.DataFrame(pareto_front, columns=['accuracy', 'model_size'])
pareto_front_df.to_csv('pareto_front_extracted.csv', index=False)

# Optionally, visualize the Pareto front
import matplotlib.pyplot as plt
plt.scatter(pareto_front_df['model_size'], pareto_front_df['accuracy'], color='red')
plt.xlabel('Model Size')
plt.ylabel('Accuracy')
plt.title('Pareto Front')
plt.show()
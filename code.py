# Import PyTorch and its neural network modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import torchvision for datasets, transforms, and pretrained models
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Import timm for additional pretrained models
import timm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # For data visualization
import sys
from tqdm.notebook import tqdm # Import timm for additional pretrained models

# Print versions of important libraries and Python
print('System version:', sys.version)
print('PyTorch version:', torch.__version__)
print('Torchvision version:', torchvision.__version__)
print('NumPy version:', np.__version__)
print('Pandas version:', pd.__version__)

# Pytorch Dataset (and Dataloader)

# Define a custom Dataset for playing cards
class PlayingCardDataset(Dataset):

    # Initialize the dataset with a directory and optional transforms
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform = transform) # Load images using ImageFolder

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.data)

    # Get a single sample by index
    def __getitem__(self, idx):
        return self.data[idx]

    # Return the class names
    @property
    def classes(self):
        return self.data.classes
    
    # Create a dataset object for the playing card images
# The images are loaded from the specified training directory
dataset = PlayingCardDataset(
    data_dir = '/kaggle/input/cards-image-datasetclassification/train'
)

len(dataset) # Get the number of samples in the dataset

# Get the image and its corresponding label from the dataset at index 1229
image, label = dataset[1229]

# Print the label of the selected image
print(label) 

# Display the image
image 

# Specify the directory containing training images
data_dir = '/kaggle/input/cards-image-datasetclassification/train'

# Convert class index to class name 
# Class number → name
target_to_class = {v : k for k, v in ImageFolder(data_dir).class_to_idx.items()}

# Show class names
print(target_to_class)

# Define a series of image transformations to apply to each image
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to 128x128 pixels
    transforms.ToTensor(), # Convert images to PyTorch tensors
])

# Path to the directory containing the training dataset
data_dir = '/kaggle/input/cards-image-datasetclassification/train'

# Create a dataset object with the specified directory and transformations
dataset = PlayingCardDataset(data_dir, transform)

# Get the 101st sample (image and label) from the dataset
image, label = dataset[100]

# Check the dimensions/shape of the image tensor
image.shape

#iterate over dataset
for image, label in dataset:
    break 

# Create a DataLoader to iterate over the dataset in batches of 32 (a group of 32 images), 
# shuffling the data each epoch
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

#iterate over the dataloader
for images, labels in dataloader:
    break

# images.shape -> Check the shape of the images tensor (e.g., batch_size, channels, height, width)
#labels.shape -> Check the shape of the labels tensor (e.g., batch_size,)

images.shape, labels.shape

# pytorch model
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes = 53):

        # Call the parent class (nn.Module) initializer
        super(SimpleCardClassifier, self).__init__()
        
        # Where we define all the parts of the model
        # Load a pretrained EfficientNet-B0 model from timm library
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True)

        # Use all layers from the base model except the last one 
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Output size from EfficientNet-B0 features
        enet_out_size = 1280
        
        # Make a classifier
        # Final layer to predict the class
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        
        # Connect these parts and return the output
        # Pass input through feature extractor
        x = self.features(x)

        # Pass extracted features to the classifier
        output = self.classifier(x)

        # Return the final predictions
        return output 
    

# Create a SimpleCardClassifier model with 53 output classes 
# and print the first 500 characters of its architecture

# Initialize the model with 53 output classes
model = SimpleCardClassifier(num_classes = 53)

# Print the first 500 characters of the model's structure
print(str(model)[:500])

# model(images) -> feed the batch of images through the model to get predictions

example_output = model(images) # feeding the batch of images through the model to get predictions
example_output.shape #[batch size, num_classes(number of classes my model is predicting)]

# training loop
# Loss function
criterion  = nn.CrossEntropyLoss()

# Define the optimizer (updates model weights to minimize the loss)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr -> learning rate

# Compute loss
# Calculate the loss between the model's predictions and the actual labels
criterion(example_output, labels)

# Show output and label dimensions
print(example_output.shape, labels.shape)

# setup datasets
# Select GPU (cuda:0) if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# Print the device being used
print(device)

# Define image preprocessing steps: resize images to 128x128 and convert them to PyTorch tensors
# Step 1: Import image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)), # - Resize images to 128x128
    transforms.ToTensor(), # - Convert images to PyTorch tensors
])

# Train = learn, 
# Validation = monitor / check progress, 
# Test = measure model accuracy on new, unseen data

# Step 2: Define dataset folder paths
# Set folder paths for training, validation, and test images
train_folder = '/kaggle/input/cards-image-datasetclassification/train'
valid_folder = '/kaggle/input/cards-image-datasetclassification/valid'
test_folder = '/kaggle/input/cards-image-datasetclassification/test'

# Step 3: Create dataset objects (connect folder + transformations)
# Create dataset objects for training, validation, and test sets
train_dataset = PlayingCardDataset(train_folder, transform = transform)
valid_dataset = PlayingCardDataset(valid_folder, transform = transform) # valid -> validation
test_dataset = PlayingCardDataset(test_folder, transform = transform)

# Create DataLoaders to load data in batches
# train_loader → learns from training data
# valid_loader → used during training to monitor performance
# test_loader → evaluates final model accuracy on completely unseen data
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True) # shuffle training data for better learning
valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False) # no shuffle for validation
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False) # no shuffle for testing

# simple training loop
# epoch - one run through the entire training dataset
num_epochs = 5 # we're going to set to train for 20 epochs

train_losses = [] # list to store training loss for each epoch
validation_losses = [] # list to store validation loss for each epoch

# for clarification - 
# train_loss = just the current epoch loss
# train_losses = list of all epoch losses

# create the model
model = SimpleCardClassifier(num_classes = 53)

# to move the model and data over CPU or GPU
model.to(device)

# we will start looping through 5 epochs
for epoch in range(num_epochs):

    # Training phase
    # set the model to train
    model.train() # Set the model to training mode
    running_loss = 0.0 # this variable will keep track of the total loss accumulated during one training epoch

    # we're gong to loop through our training dataloader to train the model
    # Loop through batches of images and labels from the training data
    for images, labels in tqdm(train_loader, desc = 'Training loop'):
        images, labels = images.to(device), labels.to(device) #put these on the CPU or GPU

        # Reset gradients before backpropagation
        optimizer.zero_grad()
        outputs = model(images) # Feed the batch of images through the model to get predictions
        
        loss = criterion(outputs, labels) # to calculate our loss for this batch
        loss.backward() #run back-propagation on the model which will update the weights in each of these steps using loss
        running_loss = running_loss + loss.item() * images.size(0) # keep track on our running loss
        
    #after one epoch (the inner loop) is done, we're going store our training loss
    train_loss = running_loss / len(train_loader.dataset) 
    train_losses.append(train_loss) # for tracking, we're going to append the train_losses's list

    # because we want to see how well the model is doing on the validation set as we train,
    # we will add a validation portion to this training loop
    
    # validation phase
    model.eval() # change the model from being in training mode to evaluation mode
    running_loss = 0.0 # track running loss in our validation dataset

    # Disable gradient calculations for validation (saves memory and speeds up)
    #just to make sure that the model weights are not touched, we will do this under torch.no_grad
    with torch.no_grad():

        # Loop through the validation dataset with a progress bar
        for images, labels in tqdm(valid_loader, desc = 'Validation loop'):
            
            # move images and labels to the device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) # Get model predictions for the batch of images
            loss =  criterion(outputs, labels) # Calculate the loss between predictions and actual labels

            # Accumulate the total loss (scaled by batch size) for later averaging
            running_loss = running_loss + loss.item() * images.size(0); 
            
    valid_loss = running_loss / len(valid_loader.dataset) # current epoch loss
    validation_losses.append(valid_loss) # append current epoch loss to the list

    # print epoch stats
    # this will show us how well we're doing by showing the training and validation loss
    print(f"Epoch  {epoch + 1} /  {num_epochs} - Train loss: {train_loss}, Validation loss: {valid_loss}")

# visualize losses
# Plot the training loss curve
plt.plot(train_losses, label = 'Training Loss') 

# Plot the validation loss curve
plt.plot(validation_losses, label = 'Validation Loss') 

# Show legend to differentiate the curves
plt.legend() 

# Add a title to the plot
plt.title("Loss over epochs") 

# Display the plot
plt.show() 

# evaluating the results
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load and preporcess the image
# Function to load and preprocess an image
def preprocess_image(image_path, transform):

    # Open the image and convert it to RGB format
    image = Image.open(image_path).convert("RGB") 

    # Apply the given transform and add a batch dimension
    return image, transform(image).unsqueeze(0)


# Predict using the model
# Function to make predictions using a trained PyTorch model
def predict(model, image_tensor, device):
    model.eval() # Set the model to evaluation mode

     # No gradient calculation (faster & uses less memory)
    with torch.no_grad():
        image_tensor = image_tensor.to(device) # Move image to CPU/GPU

        # Get raw predictions from the model
        outputs = model(image_tensor) 

        # Convert raw model outputs into probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim = 1)

    # Move to CPU & Convert to NumPy array and flatten to 1D
    return probabilities.cpu().numpy().flatten()

# Visualization
# Function to visualize predictions (image + probability bar chart)
def visualize_predictions(original_image, probablities, class_names):

    # Create a figure with 1 row and 2 columns (image and chart side-by-side)
    fig, axarr = plt.subplots(1, 2, figsize = (14, 7))

    # ---- Display the original image ----
    axarr[0].imshow(original_image) # Show the image
    axarr[0].axis("off") # Remove axis for cleaner look

    #  ---- Display the prediction probabilities ----
    axarr[1].barh(class_names, probabilities) # Horizontal bar chart
    axarr[1].set_xlabel("Probabilities") # Label for x-axis
    axarr[1].set_title("Class Predictions") # Chart title
    axarr[1].set_xlim(0, 1) # Limit x-axis from 0 to 1

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# Example usage
# Path to the test image
test_image = "/kaggle/input/cards-image-datasetclassification/test/ace of hearts/2.jpg"

# Define a sequence of transformations for the image:
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize the image to 128x128 pixels
    transforms.ToTensor() # Convert the image to a PyTorch tensor
])

# Preprocess the test image and get both the original image and the transformed tensor
original_image, image_tensor = preprocess_image(test_image, transform)

# Use the model to predict class probabilities for the input image
probabilities = predict(model, image_tensor, device)

# Get the list of class names from the dataset
class_names = dataset.classes

# Display the original image with its predicted classes and probabilities
visualize_predictions(original_image, probabilities, class_names)

# print out 10 examples randomly 
# Import the glob function to find file paths matching a pattern
from glob import glob

# Get all test images
test_images = glob('/kaggle/input/cards-image-datasetclassification/test/*/*')

# Pick 10 random images from the test set (in the test dataset)
test_examples = np.random.choice(test_images, 10)

# Loop through each example in the test dataset
for example in test_examples:

    # Preprocess the example image and get both the original and its tensor form
    original_image, image_tensor = preprocess_image(example, transform)

    # Use the model to predict class probabilities for the image tensor
    probabilities = predict(model, image_tensor, device)

    # Get the list of class names from the dataset
    class_names = dataset.classes

    # Display the original image with its predicted class probabilities
    visualize_predictions(original_image, probabilities, class_names)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Get the directory of the current script
if '__file__' in globals():
    # we are running from a script file
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # we are running from an interactive console or a script executed with -c
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Read data from CSV files using pandas
data_tsne_2 = pd.read_csv(os.path.join(current_dir,'..\\draft\\tsne_image_2_var.csv'), header=None).values
data_tsne_3 = pd.read_csv(os.path.join(current_dir,'..\\draft\\tsne_image_3_var.csv'), header=None).values

data_pca_2 = pd.read_csv(os.path.join(current_dir,'..\\draft\\pca_image_2_var.csv'), header=None).values
data_pca_3 = pd.read_csv(os.path.join(current_dir,'..\\draft\\pca_image_3_var.csv'), header=None).values

labels = pd.read_csv(os.path.join(current_dir,'..\\draft\\labels_visualizze.csv'), header=None).values

def visualize_tsne_3d(data, labels):
# Create a 3D scatter plot of the data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap=plt.cm.get_cmap('jet', 10))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('MNIST Data in 3D')

    # Create a colorbar
    cbar = plt.colorbar(scatter)
    cbar.ax.set_ylabel('Number')
    plt.show()

def visualize_tsne_2d(data, labels):
    # Plot the data
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.get_cmap('jet', 10), alpha=0.7)
    plt.colorbar()
    plt.title('t-SNE visualization of MNIST dataset')
    plt.show()

def visuzalize_pca_2d(data, labels):
    # Plot the data
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.get_cmap('jet', 10), alpha=0.7)
    plt.colorbar()
    plt.title('PCA visualization of MNIST dataset')
    plt.show()

def visuzalize_pca_3d(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap=plt.cm.get_cmap('jet', 10))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('MNIST Data in 3D')

    # Create a colorbar
    cbar = plt.colorbar(scatter)
    cbar.ax.set_ylabel('Number')
    plt.show()


# visuzalize_pca_2d(data_pca_2, labels=labels)
# visualize_tsne_2d(data_tsne_2, labels=labels)

# visuzalize_pca_3d(data_pca_3, labels=labels)
visualize_tsne_3d(data_tsne_3, labels=labels)
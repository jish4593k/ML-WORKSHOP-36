import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load a pre-trained model from Torch
model_dict = {
    "ResNet-18": models.resnet18(pretrained=True),
    "ResNet-50": models.resnet50(pretrained=True),
    "VGG-16": models.vgg16(pretrained=True),
}

selected_model = models.resnet18(pretrained=True)
selected_model_name = "ResNet-18"

selected_model.eval()

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a Tkinter GUI
root = tk.Tk()
root.title("Image Classifier")

# Variables to customize the number of top classes and display original image
num_top_classes_var = tk.IntVar(value=5)
display_original_image_var = tk.BooleanVar(value=True)

# Function to classify an uploaded image
def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = Image.open(file_path)
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = selected_model(image)
    _, predicted = output.max(1)

    class_labels = pd.read_csv("imagenet_classes.csv")
    class_name = class_labels.iloc[predicted.item()]["class"]

    conf_scores, top_classes = torch.topk(torch.softmax(output, dim=1), num_top_classes_var.get())
    top_classes = top_classes[0].numpy()
    conf_scores = conf_scores[0].numpy()

    result_label.config(text=f"Predicted Class: {class_name}")

    top_classes_text = "\n".join([f"{class_labels.iloc[i]['class']} ({conf_scores[i]:.2f})" for i in top_classes])
    top_classes_label.config(text=f"Top {num_top_classes_var.get()} Classes:\n{top_classes_text}")

    if display_original_image_var.get():
        display_image(file_path)
    else:
        img_label.config(image=None)

    visualize_confidence(output, num_top_classes_var.get())

# Function to display the uploaded image
def display_image(file_path):
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img

# Function to visualize confidence scores
def visualize_confidence(output, num_classes):
    probabilities, class_indices = torch.topk(torch.softmax(output, dim=1), num_classes)
    class_indices = class_indices[0].numpy()
    probabilities = probabilities[0].numpy()

    class_labels = pd.read_csv("imagenet_classes.csv")
    class_names = [class_labels.iloc[i]["class"] for i in class_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=probabilities, y=class_names)
    plt.xlabel("Confidence Score")
    plt.ylabel("Class Name")
    plt.title(f"Top-{num_classes} Predicted Classes")
    plt.show()

# Function to save the classification results
def save_results():
    result_text = result_label.cget("text")
    if result_text.startswith("Predicted Class:"):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(result_text)
                messagebox.showinfo("Info", "Results saved successfully!")

# Function to change the model
def change_model():
    global selected_model, selected_model_name
    model_name = model_var.get()
    selected_model = model_dict[model_name]
    selected_model_name = model_name

# Create GUI components
file_button = tk.Button(root, text="Upload Image", command=classify_image)
save_button = tk.Button(root, text="Save Results", command=save_results)
result_label = tk.Label(root, text="Predicted Class: ")
top_classes_label = tk.Label(root, text="Top Classes: ")
num_top_classes_label = tk.Label(root, text="Number of Top Classes:")
num_top_classes_entry = tk.Entry(root, textvariable=num_top_classes_var, width=3)
display_original_image_checkbox = tk.Checkbutton(root, text="Display Original Image", variable=display_original_image_var)
img_label = tk.Label(root)
model_label = tk.Label(root, text="Select Model:")
model_var = tk.StringVar()
model_var.set(selected_model_name)
model_menu = tk.OptionMenu(root, model_var, *model_dict.keys())
model_select_button = tk.Button(root, text="Change Model", command=change_model)

# Place components in the GUI
file_button.grid(row=0, column=0, padx=5, pady=5)
result_label.grid(row=1, column=0, padx=5, pady=5)
top_classes_label.grid(row=2, column=0, padx=5, pady=5)
num_top_classes_label.grid(row=3, column=0, padx=5, pady=5)
num_top_classes_entry.grid(row=3, column=1, padx=5, pady=5)
display_original_image_checkbox.grid(row=4, column=0, padx=5, pady=5)
img_label.grid(row=0, column=1, rowspan=5, columnspan=3)
save_button.grid(row=5, column=0, padx=5, pady=5)
model_label.grid(row=6, column=0, padx=5, pady=5)
model_menu.grid(row=6, column=1, padx=5, pady=5)
model_select_button.grid(row=6, column=2, padx=5, pady=5)

# Run the Tkinter main loop
root.mainloop()

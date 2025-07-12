
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from canvas import get_digit_from_canvas

# Define model architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

# Streamlit App UI
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢")
st.title("MNIST Digit Recognizer using PyTorch")

canvas_img = get_digit_from_canvas()

if canvas_img is not None:
    if st.button("🔍 Predict Digit"):
        with torch.no_grad():
            input_tensor = torch.tensor(canvas_img).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            st.subheader(f"Predicted Digit: **{pred}**")
            st.bar_chart(probs.cpu().numpy()[0])

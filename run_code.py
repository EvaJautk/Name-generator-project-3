!pip install ngrok
!pip install streamlit
!pip install pyngrok
#run authentication token 
#!ngrok authtoken 2pRoZs6LBhu0UU89o8ZQsL4vi4y_6TeUEzE6RwPnX2gRWbGTc
import subprocess

# Run ngrok authtoken command using subprocess
subprocess.run(['ngrok', 'authtoken', '2pRoZs6LBhu0UU89o8ZQsL4vi4y_6TeUEzE6RwPnX2gRWbGTc'])

# Import necessary libraries
from pyngrok import ngrok
import os

# Set up a tunnel to the Streamlit app on port 8501 (Streamlit's default port)
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at: {public_url}")

# Install Streamlit if it's not already installed (only needed for Colab)
!pip install streamlit

# Run the Streamlit app in the background using shell command
!streamlit run streamlit_app.py &  # Replace with your actual file path

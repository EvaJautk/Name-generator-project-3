#!pip install ngrok
#!pip install streamlit
#!pip install pyngrok
#run authentication token 
#!ngrok authtoken 2pRoZs6LBhu0UU89o8ZQsL4vi4y_6TeUEzE6RwPnX2gRWbGTc
#pip install pyngrok

#import subprocess

# Run ngrok authtoken command using subprocess
#subprocess.run(['ngrok', 'authtoken', '2pRoZs6LBhu0UU89o8ZQsL4vi4y_6TeUEzE6RwPnX2gRWbGTc'])

# Import necessary libraries
#from pyngrok import ngrok
#import os

# Set up a tunnel to the Streamlit app on port 8501 (Streamlit's default port)
#public_url = ngrok.connect(8501)
#print(f"Streamlit app is live at: {public_url}")

# Install Streamlit if it's not already installed (only needed for Colab)
#!pip install streamlit

# Run the Streamlit app in the background using shell command
#!streamlit run streamlit_app.py &
#subprocess.run(['streamlit', 'run', 'streamlit_app.py'])
import subprocess
import sys

# Check if pyngrok is installed, and if not, install it
try:
    import pyngrok
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])

from pyngrok import ngrok

# Now you can safely use pyngrok in your script
ngrok.set_auth_token("2pRoZs6LBhu0UU89o8ZQsL4vi4y_6TeUEzE6RwPnX2gRWbGTc")

# Open a ngrok tunnel to the Streamlit app
public_url = ngrok.connect(8501)  # Default Streamlit port
print(f"Streamlit app is accessible at: {public_url}")

# Start the Streamlit app
subprocess.run(['streamlit', 'run', 'streamlit_app.py'])

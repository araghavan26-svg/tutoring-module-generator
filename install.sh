#!/bin/bash

# Step 1: Display a message indicating the start of the setup process
echo "Setting up environment..."

# Step 2: Create virtual environment
python3 -m venv .venv

# Step 3: Activate the virtual environment
source .venv/bin/activate

# Step 4: Install dependencies from requirements.txt
pip install -r requirements.txt

# Step 5: Set up the .env file (ask the user for their OpenAI API key)
echo "Please enter your OpenAI API key:"
read OPENAI_API_KEY

# Save the API key to the .env file
echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env

# Step 6: Run the FastAPI app using uvicorn
echo "Starting the FastAPI application..."
uvicorn app.main:app --reload

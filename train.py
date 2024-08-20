import replicate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face token from the environment variables
hf_token = os.getenv('hf_token')

training = replicate.trainings.create(
  # You need to create a model on Replicate that will be the destination for the trained version.
  destination="creatorghost/Flux_With_Adi",
  version="ostris/flux-dev-lora-trainer:1296f0ab2d695af5a1b5eeee6e8ec043145bef33f1675ce1a2cdb0f81ec43f02",
  input={
    "steps": 1500,
    "hf_token": hf_token,
    "lora_rank": 16,
    "batch_size": 1,
    "hf_repo_id": "Aditya0097/Flux_With_Adi",
    "autocaption": True,
    "input_images": "./Aditya.zip",
    "trigger_word": "Aditya",
    "learning_rate": 0.0004,
    "autocaption_prefix": "Photo of Aditya\nImaging Aditya"
  },
)

#!/bin/bash

# Displaying the selection options with corresponding model names
echo "Select the config file for training:"
echo "1) TinyLlama Model (experiments/TinyLlama/config.yaml)"
echo "2) OPT Model (experiments/opt/config.yaml)"
echo "3) Phi-2 Model (experiments/phi-2/config.yaml)"

# Read user input
read -p "Enter the number (1-3) to choose the model: " choice

# Match the user's choice with the corresponding config file
case $choice in
  1)
    CONFIG_PATH="experiments/TinyLlama/config.yaml"
    MODEL_NAME="TinyLlama"
    ;;
  2)
    CONFIG_PATH="experiments/opt/config.yaml"
    MODEL_NAME="OPT"
    ;;
  3)
    CONFIG_PATH="experiments/phi-2/config.yaml"
    MODEL_NAME="Phi-2"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

# Inform the user which model is selected and the corresponding config file
echo "You have selected the $MODEL_NAME model with the config: $CONFIG_PATH"

# Run the training script with the selected config file
python trainer_hatecheck.py "$CONFIG_PATH"

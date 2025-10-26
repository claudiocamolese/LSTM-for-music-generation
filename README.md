# LSTM for music generation

This repository contains a Generative AI project focused on music generation using Long Short-Term Memory (LSTM) networks.  

**LSTMs** are a special type of Recurrent Neural Network (RNN) designed to overcome the vanishing gradient problem commonly encountered in traditional RNNs.  
Unlike standard RNNs or Hidden Markov Models, LSTMs can effectively capture long-term dependencies, making them ideal for sequence-based tasks such as text, speech and music generation.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/7d45abfe-0e9b-49c2-a465-9e1d96c2e7c9" width="500">
</p>

An LSTM unit is composed of a cell and three gates (input, forget and output) that control how information flows through the network:  
- Cell: acts as long-term memory, storing information over many timesteps.  
- Forget gate: decides which information from the previous state should be discarded.  
- Input gate: determines what new information should be added to the current state.  
- Output gate: controls which parts of the current state are used to produce the next output.  

Through this selective control mechanism, LSTMs can retain meaningful patterns over long sequences, enabling the model to learn melodic structure, rhythm, and harmony from musical data.  

---


## How to run

All parameters for training, evaluation, generation and tracking can be set in `config.yaml`.  
The code supports the *Comet ML* framework for tracking experiments—make sure to set the flags for training and tracking to `True`.  

If you don’t have a GPU and want to use *Colab*, run:

```python
!cp /content/drive/MyDrive/music_lstm/content/
!pip install -r music_lstm/requirements.txt
!python music_lstm/main.py
```
---

## Datasets
The code supports any dataset in .abc format.
Pre-trained models are provided for the *Irish* song and *Pop* song datasets, with weights saved in `./models/`.

Additionally, the code allows you to merge multiple datasets to create a custom music genre. Have fun exploring and generating your own unique compositions!

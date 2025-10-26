# LSTM for music generation

This repository contains a Generative AI project focused on music generation using Long Short-Term Memory (LSTM) networks.  

**LSTMs** are a special type of Recurrent Neural Network (RNN) designed to address the vanishing gradient problem commonly encountered in traditional RNNs.  
Unlike standard RNNs or Hidden Markov Models, LSTMs can effectively capture long-term temporal dependencies, making them ideal for sequence-based tasks such as text, speech and music generation.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/7945efaf-f8d8-4928-89b0-ba377166153a" width="400">
</p>

An LSTM unit is composed of a cell and three gates (input, forget and output) that control how information flows through the network:  
- Cell: acts as long-term memory, storing information over many timesteps.  
- Forget gate: decides which information from the previous state should be discarded.  
- Input gate: determines what new information should be added to the current state.  
- Output gate: controls which parts of the current state are used to produce the next output.  

Through this selective control mechanism, LSTMs can retain meaningful patterns over long sequences, enabling the model to learn melodic structure, rhythm, and harmony from musical data.  

---


## How to run

In `config.yaml` you can specify all your parameters for training, evaluation, generation and tracking. The code supports the *comet_ml* framework for tracking the training of the model.
Remember to set the flag to `True` both for training and tracking. 

If you don't have GPU and you want to use *Colab* use:
```python
!cp /content/drive/MyDrive/music_lstm /content/
!pip install -r music_lstm/requirements.txt
!python music_lstm/main.py
```
---

## Datasets
The code supports all type of .abc format datasets. Moreover, the model is trained both on `irish_song` and `pop_song` dataset. Both weights are saved in the `./models/`. 
In addition, the code also supports the possibility to create a custom dataset mergind two differents dataset to be able to create your new music genre. Have Fun!!

import torch
import numpy as np
import os
from scipy.io.wavfile import write


def generate(model, start_sequence, vocab, max_length=100, output_path='./outputs/', experiment = None, device='cpu'):
    """Functions called during the inference of the model

    Args:
        model (torch.model): model to use for generation
        start_sequence (str): special token for LSTM to indicate the beginning of the generation. Helps the model to undestrand where it need to start the generation
        vocab (dict): sequence of admitted tokens 
        max_length (int, optional): Max token lenght to undestrand generation. Defaults to 100.
        output_path (str, optional): where to upload the generated file. Defaults to './outputs/'.
        experiment (comet_ml.experiment, optional): track the training. Defaults to None.

    Returns:
        str: content of the generated txt
    """
    model.to(device)
    model.eval()

    # inverse dictionary to pass from num to char
    idx_to_char = {idx: char for char, idx in vocab.items()}
    char_to_idx = vocab

    # conversion computation
    input_seq = [char_to_idx[char] for char in start_sequence]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device) # dimension [1, seq_len]

    # initialize hidden state (h_0, c_0)
    hidden = model.init_hidden(1) # creates an hidden state (num_layers, batch_size=1, hidden_dim)

    generated_sequence = start_sequence
    
    """
    Inference:
        - loop until max_lenght is reached
        - LSTM returns output (logits predicted for each token), hidden state (h_n, c_n)
        - take the last token in the prediction, converts logits in probabilities
        - randomly choose next token with the computed probabilities.
            In the text/music generation, we dont want to take always the token with the highest probability, otherwise it will be determinist.
            Adding a probability increase creativity in the model generation
        - convert idx to char
        - added generated char to generated_sequence 
    """

    with torch.no_grad():
        for _ in range(max_length):
            # Find next token
            output, hidden = model(input_tensor, hidden) # ouput = [batch, seq_len, vocab_size]
            prob = torch.nn.functional.softmax(output[-1], dim=0).cpu().numpy()
            next_idx = np.random.choice(len(prob), p=prob)
            
            # Convert 
            next_char = idx_to_char[next_idx]
            generated_sequence += next_char
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'generated_music.txt')
    with open(output_file, 'w') as f:
        f.write(generated_sequence)
    
    print(f"✅ Generated text file saved in {output_file}")

    if experiment:
        experiment.flush()
    return generated_sequence
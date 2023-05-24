import torch
import torchaudio
import gradio as gr

from cnn import CNNetwork
from preprocess import process_raw_wav, _wav_to_spec

model = CNNetwork()
state_dict = torch.load('models/deeptruth_20230524_013021.pth')
model.load_state_dict(state_dict)

LABELS = ["clone", "pure"]


def greet(mic_input, upload_input):
    if mic_input:
        sr, wav = mic_input
    elif upload_input:
        sr, wav = upload_input
    else:
        raise gr.Error("There needs at least one input.")
    
    print(f"WAV {wav}")

    wav = torch.tensor([wav]).float()
    wav = process_raw_wav(wav, sr, 48000, 5)
    wav = _wav_to_spec(wav, 48000)

    print(f"WAV {wav}")

    model_input = wav.unsqueeze(0)
    output = model(model_input)
    print(output)

    prediction_index = torch.argmax(output, 1).item()


    if LABELS[prediction_index] == "clone":
        return "That is likely voice cloned audio."
    else:
        return "That is likely pure, non-cloned audio."

demo = gr.Interface(fn=greet, inputs=["mic", "audio"], outputs="text")

demo.launch(server_port=3000) 
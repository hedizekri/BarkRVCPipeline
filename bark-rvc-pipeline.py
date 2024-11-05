import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

## Bark imports
import nltk
import numpy as np
from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav

# RVC imports
from rvc_infer import rvc_convert

#Environment variables
SCRIPT = "Hello, my name is Hedi. I am using Bark and RVC in a pipeline to create realistic voices out of text."
AUDIO_FILENAME = "bark_generated_audio.wav"
RVC_INPUT_DIRECTORY = "input"
RVC_MODEL_NAME = "v2/fr_speaker_6"  # Voices library here : https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
WORD_COUNT_THRESHOLD = 85  # If input text words count is larger than this value, we use a specific audio generation process



def count_nb_words(text):  
    return len(nltk.word_tokenize(text))  # Return the words count in the script

def audio_generation_short_text(text, rvc_model_name):
    audio_array = generate_audio(text, history_prompt=rvc_model_name)  # Audio generation process for short input texts
    return audio_array

def audio_generation_long_text(text, rvc_model_name):
    text_sent = nltk.sent_tokenize(text)
    silence = np.zeros(int(0.1 * SAMPLE_RATE))  # quarter second of silence
    pieces = []
    i = 0

    while i < len(text_sent):
        if i + 1 < len(text_sent):  # If there is a next sentence
            combined_sentence = text_sent[i] + " " + text_sent[i + 1]
            i += 2  # Increment by 2 as we are taking two sentences
        else:
            combined_sentence = text_sent[i]
            i += 1  # Increment by 1 as only one sentence is left

        print(combined_sentence)
        audio_array = generate_audio(combined_sentence, history_prompt=rvc_model_name)
        pieces += [audio_array, silence.copy()]
    return np.concatenate(pieces)

def generate_audio_from_text(script, rvc_model_name, audio_filename, rvc_input_directory, word_count_threshold):
    # download and load all models
    preload_models()

    nb_words = count_nb_words(script)
    print(nb_words)

    # Choose the right text sampling process to generate audio depending on the text length
    audio = []
    if nb_words < word_count_threshold:
        audio = audio_generation_short_text(script, rvc_model_name)
    else: 
        audio = audio_generation_long_text(script, rvc_model_name)

    # Generate the filename with the counted number
    rvc_input_filepath = f"{rvc_input_directory}/{audio_filename}"

    # Write the wav file
    write_wav(rvc_input_filepath, SAMPLE_RATE, audio)

    #Print to validate
    print(f"Bark audio generated : {rvc_input_filepath}")

    # RVC inference
    rvc_convert(model_path=f"models/{rvc_model_name}.pth",
                input_path=rvc_input_filepath)

if __name__ == "__main__":
    generate_audio_from_text(SCRIPT, RVC_MODEL_NAME, AUDIO_FILENAME, RVC_INPUT_DIRECTORY, WORD_COUNT_THRESHOLD)

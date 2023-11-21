import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

## CSV import
import csv

## Bark imports
import nltk  # we'll use this to split into sentences
import numpy as np
from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav

# RVC imports
import sounddevice as sd
import soundfile as sf
from rvc_infer import rvc_convert

#Environment variables
MALE_SPEAKER = "v2/fr_speaker_6"
FEMALE_SPEAKER = "v2/fr_speaker_1"

RVC_INPUT_DIRECTORY = "input"

MALE_MODELS_LIST = ['ascuns', 'asterion', 'bigflo', 'davidk', 'farod', 'hardisk', 'itachi', 'micmaths', 'rayton', 'theodort']
FEMALE_MODELS_LIST = ['alizee', 'angele', 'lena', 'margotrobbie']

SCRIPT_CSV_PATH = "/Users/hedizekri/AI/fantasy_ai/models/fantasy1/fantasy_1_output/lora_fantasy_1_output.csv"

def count_nb_words(text):
    return len(nltk.word_tokenize(script))

def audio_from_word(text):
    audio_array = generate_audio(text, history_prompt=FEMALE_SPEAKER)
    return audio_array

def audio_from_sent(text):
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
        audio_array = generate_audio(combined_sentence, history_prompt=MALE_SPEAKER)
        pieces += [audio_array, silence.copy()]
    return np.concatenate(pieces)

def extract_sent_from_csv(csv_path):
    data_list = [] # Liste pour stocker les données

    with open(csv_path, 'r') as file: # Ouverture et lecture du fichier CSV
        csv_reader = csv.reader(file)
        next(csv_reader)  # Ignorer l'en-tête si nécessaire

        for row in csv_reader:
            data_list.append(row[0])

    return data_list

if __name__ == "__main__":
    # download and load all models
    preload_models()

    # Initialize variables
    scripts_list = extract_sent_from_csv(SCRIPT_CSV_PATH)
    script = scripts_list[0]

    nb_words = count_nb_words(script)
    print(nb_words)

    # Choose the right text sampling process to generate audio depending on the text length
    audio = []
    if nb_words < 85:
        audio = audio_from_word(script)
    else: 
        audio = audio_from_sent(script)

    # Save audio to disk
    # Count the number of files in the "results" directory
    #num_files = len([f for f in os.listdir(RVC_INPUT_DIRECTORY) if os.path.isfile(os.path.join(RVC_INPUT_DIRECTORY, f))])

    # Generate the filename with the counted number
    #audio_filename = f"{RVC_INPUT_DIRECTORY}/generated_audio_{num_files}.wav"
    audio_filename = "generated_audio_test.wav"
    rvc_input_filepath = f"{RVC_INPUT_DIRECTORY}/{audio_filename}"
    rvc_output_filepath = f"{RVC_OUTPUT_DIRECTORY}/{audio_filename}"

    # Write the wav file
    write_wav(rvc_input_filepath, SAMPLE_RATE, audio)

    #Print to validate
    print(f"File created : {rvc_input_filepath}")

    # RVC inference
    model_name = MALE_MODELS_LIST[2]
    rvc_convert(model_path=f"models/{model_name}.pth",
                f0_up_key=0,
                input_path=rvc_input_filepath)

    


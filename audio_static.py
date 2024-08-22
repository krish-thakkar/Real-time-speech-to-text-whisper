import argparse
import os
import numpy as np
import whisper
import torch
import speech_recognition as sr

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--audio_file", required=True, help="Path to the audio file", type=str)
    args = parser.parse_args()

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    # Load the audio file using SpeechRecognition
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(args.audio_file)
    with audio_file as source:
        audio_data = recognizer.record(source)
    
    # Convert the audio data to a numpy array
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

    # Transcribe the audio
    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
    text = result['text'].strip()

    # Print the transcription
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Transcription:")
    print(text)


if __name__ == "__main__":
    main()

# DO NOT RUN THIS SCRIPT WITHOUT GPU

import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

# os.system("pip install TTS boto3 cutlet fugashi unidic-lite")
# os.system("apt-get install -y espeak-ng mecab libmecab-dev mecab-ipadic-utf8")

# shell commands are commented

# !pip install TTS
# !apt-get install -y espeak-ng
# !pip install boto3
# !pip install cutlet
# !apt-get install mecab libmecab-dev mecab-ipadic-utf8 -y
# !pip install fugashi cutlet unidic-lite


os.system("pip install TTS boto3 cutlet fugashi unidic-lite")
from TTS.api import TTS

os.system("apt-get install -y espeak-ng mecab libmecab-dev mecab-ipadic-utf8")
import torch
import boto3
device = "cuda" if torch.cuda.is_available() else "cpu"

aws_access_key_id = ACCESS_KEY_ID
aws_secret_access_key = SECRET_ACCESS_KEY_ID
region_name = "us-east-1"

s3 = boto3.client('s3',
                  aws_access_key_id = aws_access_key_id,
                  aws_secret_access_key = aws_secret_access_key,
                  region_name = region_name)

def load_speaker(genre):
  bucket_name = 'storyfromimages'
  file_key = f"speakers/{genre}_speaker.mp3"

  try:
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    audio_data = response['Body'].read()

    return audio_data
  except Exception as e:
    print(f"Error fetching audio file for genre '{genre}'")
    return None

def upload_to_s3(file_path, s3_file_key):
  bucket_name = 'storyfromimages'
  full_s3_file_key = f"output-audio/{s3_file_key}"
  try:
    s3.upload_file(file_path, bucket_name, full_s3_file_key)
    s3.put_object_acl(ACL='public-read', Bucket=bucket_name, Key=full_s3_file_key)
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{full_s3_file_key}"
    print(f"File uploaded to s3 : {s3_url}")
    return s3_url
  except Exception as e:
    print(f"Error uploading file: {e}")
    return None

def generate_speech(stories, genre):
  # fetch the speaker from load_speaker
  speaker_audio = load_speaker(genre)

  # initialize the tts model (xtts_v2)
  tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

  #iterate over each story and generate speech for them
  for lang,story in stories.items():
    # file_path is the audio file which will be created in the drive by the model
    file_path = f"xtts_test_{lang}.wav"
    # s3_file_key is the audio file name in s3 storage in output-audio folder
    s3_file_key = f"{lang}-output.wav"
    # generate audio and store it in drive
    tts.tts_to_file(text=story, file_path=file_path, speaker_wav=speaker_audio, language=lang)
    # upload the audio file to s3
    upload_to_s3(file_path, s3_file_key)

# generate_speech(text, 'fantasy')

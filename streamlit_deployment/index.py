import os

# Install dependencies 
os.system("pip install streamlit transformers TTS boto3 cutlet fugashi unidic-lite")
os.system("apt-get install -y espeak-ng mecab libmecab-dev mecab-ipadic-utf8")

import streamlit as st
import time
from PIL import Image


##-----------------------##

# Image Captioning

##------------------------##

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_captions(image_paths):
  # list to store the images to feed the model
  images = []
  # iterate through each path and store the image in the list
  for path in image_paths:
    image = Image.open(path).convert("RGB")
    images.append(image)
  # list to store captions
  captions = []
  # iterate over each image, feed it to the model and generate caption for it
  for image in images:
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions.append(caption)
  # return the captions list
  return captions

##------------------------------##

# Story Generation 

##------------------------------##

from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_story(captions, genre):
  try:
    # call the gemini 2.0 flash api
    response = client.models.generate_content(
      model = "gemini-2.0-flash",
      contents = f"""
      I have multiple images.
      Generate a complete {genre} story of around 450 words by fictionally linking these images.
      The story should be logical and should connect all the characters.
      I will give you the description of each image as a list of captions (each caption represents an image).
      The response should contain only the story with its title.
      Dont include any other stuff in the story(such as where you have included the caption, etc.)
      Also, the first line will contain only the title of the story (no 'Title:' or '##' anything like this should precede, only the text).
      \n{captions}
      """
    )
    # store the response in story
    story = response.text.strip()
    if story:
      return story
    else:
      print("Story generation failed")
      return None
  except Exception as e:
    print(f"Error while translating: {e}")
    return None

##------------------------------##

# Language Translation

##------------------------------##

def translate(text, target_lang):
  # we have 5 languages which are stored in a dictionary, the keys being their code which the xtts_v2(tts) model understands
  languages = {
      'hi': 'hindi',
      'es': 'spanish',
      'ja': 'japanese',
      'de': 'german'
  }
  try:
    # call the gemini 2.0 flash to translate the story
    response = client.models.generate_content(
      model = "gemini-2.0-flash",
      contents = f"Convert the following story from english to {languages[target_lang]} while preserving context and style, and give me only the converted story as output:\n\n{text}"
    )
    # store the translated_text
    translated_text = response.text.strip()

    if translated_text:
      return translated_text
    else:
      print(f"Translation for {target_lang} failed")
      return None
  except Exception as e:
    print(f"Error while translating to {target_lang}: {e}")
    return None


# function which will take the english story as input and return a dictionary of stories, their value being the translated version of each.
# the tts model will call this function to store and generate speech in multiple languages

def translate_all(text):
  langs = ['hi', 'es', 'ja', 'de']
  stories = {}
  # the given argument is the english story itself
  stories['en'] = text
  # iterate over all the other languages and fill the dictionary
  for lang in langs:
    # call the translate function to convert the story from english to the target language
    stories[lang] = translate(text, lang)
  # return stories
  return stories

##--------------------------------##

# Text to speech 

##---------------------------------##

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

from TTS.api import TTS

import torch
import boto3
device = "cuda" if torch.cuda.is_available() else "cpu"

aws_access_key_id = AWS_ACCESS_KEY
aws_secret_access_key = AWS_SECRET_ACCESS_KEY
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

def upload_to_s3(file_path, s3_file_key, upload_time):
  bucket_name = 'storyfromimages'
  full_s3_file_key = f"output-audio/{upload_time}_{s3_file_key}"
  try:
    s3.upload_file(file_path, bucket_name, full_s3_file_key)
    s3.put_object_acl(ACL='public-read', Bucket=bucket_name, Key=full_s3_file_key)
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{full_s3_file_key}"
    print(f"File uploaded to s3 : {s3_url}")
    return s3_url
  except Exception as e:
    print(f"Error uploading file: {e}")
    return None

def generate_speech(stories, genre, upload_time):
  # fetch the speaker from load_speaker
  speaker_audio = load_speaker(genre)

  # initialize the tts model (xtts_v2)
  tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

  #iterate over each story and generate speech for them
  for lang,story in stories.items():
    # file_path is the audio file which will be created in the drive by the model
    file_path = f"xtts_test_{lang}.wav"
    # s3_file_key is the audio file name in s3 storage in output-audio folder
    s3_file_key = f"{lang}.wav"
    # generate audio and store it in drive
    tts.tts_to_file(text=story, file_path=file_path, speaker_wav=speaker_audio, language=lang)
    # upload the audio file to s3
    upload_to_s3(file_path, s3_file_key, upload_time)

##-----------------------------------##

def download_and_play_audio(upload_time):
  try:
    response = s3.list_objects_v2(Bucket="storyfromimages", Prefix="output-audio/")

    languages = ['en','hi','es','ja','de']
    audio_files = {}
    language_code = {
      'en': 'English',
      'hi': 'Hindi',
      'es': 'Spanish',
      'ja': 'Japanese',
      'de': 'German'
  }
    for obj in response.get('Contents', []):
      file_key = obj['Key']
      for lang in languages:
        if f"{upload_time}_{lang}" in file_key:
          audio_files[lang] = file_key
    
    if len(audio_files) == 5:
      for lang, file_key in audio_files.items():
        language = language_code[lang]
        local_filename = f"{lang}_audio.wav"
        s3.download_file('storyfromimages', file_key, local_filename)

        st.write(f"{language} Audio:")
        st.audio(local_filename, format="audio/wav")
        with open(local_filename, 'rb') as audio_file:
          st.download_button(label=f"Download {lang.upper()} Audio", data=audio_file, file_name=f"{lang}_audio.wav", mime="audio/wav")
    else:
      st.error(f"Error: Could not find exactly 5 audio files for {upload_time} upload time, {len(audio_files)}")

  except Exception as e:
    st.error(f"Error downloading audio: {e}")

def save_uploaded_files(uploaded_files):
    image_paths = []
    save_dir = "uploaded_images"
    os.makedirs(save_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)

    return image_paths

def create_stories(image_paths, genre):
    captions = generate_captions(image_paths)
    story = generate_story(captions, genre)
    stories = translate_all(story)  # Returns dict {'en': story_en, 'hi': story_hi, ...}
    return stories

def render_english_story(story):
    st.subheader("Generated Story")
    st.write(story)

def render_audio(stories, genre, upload_time):
    generate_speech(stories, genre, upload_time)

def main():
    st.title("StoryLens: An AI Storyteller")

    with st.form(key='story_form'):
        uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
        if uploaded_files:
          st.write("Uploaded Images:")
          cols = st.columns(3)

          for index,file in enumerate(uploaded_files):
            img = Image.open(file)
            with cols[index % 3]:
              st.image(img, caption=f"Image {index+1}", use_container_width=True)
        
        genre = st.selectbox("Select a Genre", ['horror', 'romantic', 'fantasy', 'psycho_thriller', 'crime_fiction'])
        submit_button = st.form_submit_button(label='Generate Story and Audio')


    if submit_button:
        upload_time = time.time()
  
        if uploaded_files and genre:
            image_paths = save_uploaded_files(uploaded_files)

            with st.spinner("Please wait while we generate a story..."):
              stories = create_stories(image_paths, genre)
            render_english_story(stories['en'])

            with st.spinner("Please wait while we generate audio files..."):
              render_audio(stories, genre, upload_time)
            download_and_play_audio(upload_time)

        else:
            st.error("Please upload images and select a genre.")
    

if __name__ == "__main__":
    main()


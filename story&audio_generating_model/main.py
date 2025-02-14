# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YMVry-A5lEsSXf0BWrj3_07qnQoHKUI9
"""

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os
os.chdir("/content/drive/MyDrive/story-from-images-project")

from image_captioning_module import generate_captions
from story_generation_module import generate_story
from language_conversion_module import translate_all
from final_tts_module import generate_speech

# we have three functions, 1 which creates stories, and the other generates text and audio output. This is to render the text output faster
# 1. create_stories() , 2. render_audio() , 3. render_english_story()

def create_stories(image_paths, genre):
  # From the backend , receive the images as a list of url and the selected genre

  # Step 1 : Generate the caption for each image
  captions = generate_captions(image_paths)

  # Step 2 : Generate story based on the captions and selected genre
  story = generate_story(captions, genre)

  # Step 3 : Do contextual translation the english story to other languages for speech generation
  # This function returns a dictionary having key as language and its value as the story in that language
  stories = translate_all(story)

  return stories

def render_english_story(story):
  # This step is our first output i.e english text
  print(story)

def render_audio(stories, genre):
  # Step 4 : Generate speech for all languages according to the selected genre
  generate_speech(stories, genre)
  # This function creates audio files for all 5 langauges and stores them both locally and in s3

# image_paths and genre will be fetched from the backend and passed to the function
image_paths = ["test-image-1.jpg", "test-image-2.jpg"]
genre = 'romantic'

stories = create_stories(image_paths, genre)

render_english_story(stories['en'])

render_audio(stories,genre)

# After this pass the stories to the backend which will be then displayed in the frontend
# for lang, story in stories.items():
#   print(f"{lang}: {story}", end='\n')

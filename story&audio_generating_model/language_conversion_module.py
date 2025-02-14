# we use gemini 2.0 flash to contextually translate the english story to multiple languages
from google import genai
client = genai.Client(api_key="AIzaSyAh3jABypOv6effFX_wh1iEmcIsOExm7Ks")

# function to translate the english story to any target language
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

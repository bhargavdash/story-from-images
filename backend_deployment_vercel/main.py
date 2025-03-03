from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.oauth2 import service_account
import base64
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowed frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)
service_account_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
credentials = service_account.Credentials.from_service_account_file(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

client = genai.Client(
    credentials=credentials,
    api_key=None,  # Set to None when using service account
    project="storyfromimages",
    location="us-central1",
    vertexai=True  # Explicitly set this to True
)

@app.post("/generate_story/")
async def generate_story(genre: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        print("Story generation started...")
        image_parts = []
        for i,file in enumerate(files):
            image_bytes = await file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            image_parts.append(
                types.Part.from_bytes(
                    data = base64.b64decode(base64_image),
                    mime_type = "image/jpeg",
                )
            )
        
        text_part = types.Part.from_text(text=f"""I want you to generate a fictional {genre} story in about 500 words based on the images that i provide you. Extract details from each of the images, look for the objects, background etc. Also check if the same people are present in multiple images. 
        Finally generate a story with a nice title (end it with a full stop) as the only output, nothing else.
        """)

        model = "gemini-2.0-flash-001"
        contents = [
            types.Content(
            role="user",
            parts= image_parts + [text_part]
          )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 8192,
            response_modalities = ["TEXT"],
        )
        full_response = ""
        for chunk in client.models.generate_content_stream(
            model = model,
            contents = contents,
            config = generate_content_config,
            ):
                full_response += chunk.text
        
        return full_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

def translate_story(story: str, lang: str):
    """Translate the story into a specified language using Gemini."""
    try:
        languages = {
            "hi" : "Hindi",
            "es" : "Spanish",
            "de" : "German",
            "ja" : "Japanese"
          }

        text1 = types.Part.from_text(text=f"""I will provide you a story with a title below. I want you to contextually translate this story into {languages[lang]} language. 
            {story}
            Give me the translated story as the only output""")

        model = "gemini-2.0-flash-001"
        contents = [
          types.Content(
               role="user",
               parts=[text1]
          )
        ]
        generate_content_config = types.GenerateContentConfig(
          temperature = 1,
          top_p = 0.95,
          max_output_tokens = 8192,
          response_modalities = ["TEXT"],
        )
        response = ""
        for chunk in client.models.generate_content_stream(
          model = model,
          contents = contents,
          config = generate_content_config,
        ):
            response += chunk.text
        return response
    except Exception as e:
        raise Exception(f"Error translating story to {lang}: {str(e)}")


@app.post("/generate_audio/")
def generate_audio(story: str = Form(...), genre: str = Form(...), upload_time: str = Form(...)):
    try:
        print("Generating audio...")
        stories = {}
        langs = ["hi", "es", "de", "ja"]
        stories["en"] = story
        print("Translating text...")
        for lang in langs:
            translated_story = translate_story(story, lang)
            stories[lang] = translated_story
            print(f"Translated {lang} text")
        
        print("Story translation complete")
        
        print("Audio generation started...")    
        tts_stories = {
            "stories" : stories,
            "genre" : genre,
            "upload_time" : upload_time
        }

        tts_api_url = "https://bhargavdash--tts-model-api-api.modal.run/generate_speech"
        headers = {
            "Content-Type" : "application/json"
        }
        response = requests.post(tts_api_url, json=tts_stories, headers=headers)
        response.raise_for_status()
        audio_urls = response.json()
        return audio_urls

    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"TTS API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)

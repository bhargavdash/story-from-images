from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from google.oauth2 import service_account
import base64
import requests
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowed frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Authentication setup
try:
    # For render deployment
    service_account_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Initialize Vertex AI
    vertexai.init(
        project="storyfromimages",
        location="us-central1",
        credentials=credentials
    )
except Exception as e:
    # Fallback for local development
    try:
        credentials = service_account.Credentials.from_service_account_file(
            "service_account.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Initialize Vertex AI
        vertexai.init(
            project="storyfromimages",
            location="us-central1",
            credentials=credentials
        )
    except Exception as local_e:
        print(f"Failed to initialize with service account file: {local_e}")
        # Let it continue without credentials, which will use application default credentials
        # for local development environment
        vertexai.init(
            project="storyfromimages",
            location="us-central1"
        )

@app.post("/generate_story/")
async def generate_story(genre: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        print("Story generation started...")
        image_parts = []
        
        for file in enumerate(files):
            image_bytes = await file[1].read()
            image_parts.append(
                Part.from_data(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            )
        
        text_prompt = f"""I want you to generate a fictional {genre} story in about 500 words based on the images that i provide you. Extract details from each of the images, look for the objects, background etc. Also check if the same people are present in multiple images. 
        Finally generate a story with a nice title (end it with a full stop) as the only output, nothing else.
        """
        
        text_part = Part.from_text(text_prompt)
        
        # Initialize the model
        model = GenerativeModel("gemini-2.0-flash-001")
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192
        )
        
        # Combine parts and generate content
        responses = model.generate_content(
            contents=[*image_parts, text_part],
            generation_config=generation_config,
            stream=True
        )
        
        full_response = ""
        for response in responses:
            if hasattr(response, 'text'):
                full_response += response.text
        
        return full_response
    except Exception as e:
        print(f"Error in generate_story: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

def translate_story(story: str, lang: str):
    """Translate the story into a specified language using Gemini."""
    try:
        languages = {
            "hi": "Hindi",
            "es": "Spanish",
            "de": "German",
            "ja": "Japanese"
        }
        
        prompt = f"""I will provide you a story with a title below. I want you to contextually translate this story into {languages[lang]} language. 
            {story}
            Give me the translated story as the only output"""
        
        # Initialize the model
        model = GenerativeModel("gemini-2.0-flash-001")
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192
        )
        
        # Generate translation
        responses = model.generate_content(
            contents=prompt,
            generation_config=generation_config,
            stream=True
        )
        
        translated_text = ""
        for response in responses:
            if hasattr(response, 'text'):
                translated_text += response.text
        
        return translated_text
    except Exception as e:
        print(f"Error in translate_story: {str(e)}")
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
            "stories": stories,
            "genre": genre,
            "upload_time": upload_time
        }

        tts_api_url = "https://bhargavdash--tts-model-api-api.modal.run/generate_speech"
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(tts_api_url, json=tts_stories, headers=headers)
        response.raise_for_status()
        audio_urls = response.json()
        
        return audio_urls
    except requests.RequestException as e:
        print(f"TTS API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS API request failed: {str(e)}")
    except Exception as e:
        print(f"Error in generate_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Story generation API is running"}

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)

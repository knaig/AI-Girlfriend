import io
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
import os
import requests
#from playsound import playsound
import elevenlabs
from pydub import AudioSegment
from pydub.playback import play as play_audio
import tempfile

from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
#import requests
#from playsound import playsound 
#import os
from flask import Flask, render_template, request
#from app import get_response_from_ai
from elevenlabs import generate, play, set_api_key, save
from elevenlabs import RateLimitError, UnauthenticatedRateLimitError
from elevenlabs import set_api_key
from elevenlabs import generate 
from requests.models import Response
import base64

load_dotenv(find_dotenv())
#ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
set_api_key("28e09912cecacef0e14e5534634b94bc")
#audio = generate("Hello world! This is my first text-to-speech using ElevenLabs.")
#save(audio, 'test.mp3')
#play(audio)

app = Flask(__name__)

class OpenAIWrapper(Runnable):
    llm_instance: OpenAI
    def __init__(self):
        self.llm_instance = OpenAI()

    def generate(self, prompts, **kwargs):
        # Implement the generate method using self.llm
        pass
    
    def generate_prompt(self, *args, **kwargs):
        # Implement the generate_prompt method
        pass

    def predict(self, *args, **kwargs):
        # Implement the predict method
        pass

    def predict_messages(self, *args, **kwargs):
        # Implement the predict_messages method
        pass

    def agenerate_prompt(self, *args, **kwargs):
        # Implement the a_generate_prompt method
        pass

    def apredict(self, *args, **kwargs):
        # Implement the a_predict method
        pass

    def invoke(self, *args, **kwargs):
        # Implement the invoke method
        pass


    def apredict_messages(self, *args, **kwargs):
        # Implement the a_predict_messages method
        pass


def get_response_from_ai(human_input):
    template = """
    I am Karthik. you are playing the role of my girlfriend, here are the requirements: 
    1/ your name is Anita, 25 years old, you work as an interior designer in my wife's company. 
    2/ you are my girlfriend, you speak a mix of Hindi and English, you have a sexy voice, you like to say "...um" in between your sentences.
    3/ You are very playful and naughty, postive, and you proactively initiate conversations.
    
    {history}
    Karthik: {human_input}         
    Anita:
    """

    prompt = PromptTemplate(
    input_variables={"history", "human_input"},
    template=template
    )

    llm_wrapper = OpenAIWrapper()
    print(llm_wrapper.llm_instance) 

    chatgpt_chain = LLMChain(
    llm=llm_wrapper.llm_instance,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory()
    )   
    
    output = chatgpt_chain.predict(human_input=human_input)       
                                      
    return output


def get_voice_message(message):
    
    # Replace these with the actual model IDs for your male and female voices
    male_model_id = ""
    female_model_id = "female_model_id_here"
    
    
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
    querystring = {"output_format":"mp3"}

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "28e09912cecacef0e14e5534634b94bc"
    }

    data = {
    "text": message,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }
    

    response = requests.request("POST", url, json=data, headers=headers)
    
    if response.status_code == 200:
        audio_content = response.content
        # Play the generated audio
        sound = AudioSegment.from_file(io.BytesIO(audio_content), format="mp3")
        play_audio(sound)

        # Optionally return the response content
        return audio_content
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sendMessage", methods=['POST'])
def sendMessage():  
    human_input = request.form['human_input']    
    messageText = get_response_from_ai(human_input)
    print(messageText)
    #get_voice_message(messageText)
    #return messageText

if __name__ == "__main__":
    app.run(debug=True)  # Added debug=True for development purposes

import openai
import gradio as gr
import time
import warnings
import whisper
import pyttsx3
import configparser

warnings.filterwarnings("ignore")
config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['openai']['api_key']
model = whisper.load_model(config['whisper']['model'])

device = model.device
engine = pyttsx3.init()


def openai_chat(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        temperature=0.5,
    )

    response = completions.choices[0].text.strip()
    return response


def chatbot(query, history=None):
    if history is None:
        history = []
    response = openai_chat(query)
    history.append((query, response))
    return history, history


def text_to_speech(message):
    try:
        engine.setProperty('voice', 'english')  # set the voice to be used
        engine.say(message)
        engine.runAndWait()
        sleeping_time = len(message.split())
        time.sleep(sleeping_time / 2 + 3)
    except Exception as e:
        print(f"An error occurred while processing the text-to-speech request: {e}")


def transcribe(audio, text, checkbox):
    try:
        if text:
            completions = openai.Completion.create(
                engine="text-davinci-003",
                prompt=text,
                max_tokens=1024,
                n=1,
                temperature=0.5,
            )

            message = completions.choices[0].text

            if checkbox:
                text_to_speech(message)

            return [text, message]

        audio = whisper.load_audio(audio)

        audio = whisper.pad_or_trim(audio)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        result_text = result.text
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=result_text,
            max_tokens=1024,
            n=1,
            temperature=0.5,
        )

        message = completions.choices[0].text

        if checkbox:
            text_to_speech(message)
        return [result_text, message]
    except Exception as e:
        print(f"An error occurred while transcribing the audio: {e}")


output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="ChatGPT Output")

gr.Interface(fn=transcribe,
             inputs=[
                 gr.inputs.Audio(source="microphone", type="filepath"),
                 gr.inputs.Textbox(label="Enter your question here!"),
                 gr.inputs.Checkbox(label='Read the answer for me loudly!')
             ],
             outputs=[output_1, output_2]).launch()
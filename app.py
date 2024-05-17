import os 
import wave
import numpy as np
import pyaudio
from scipy.io import wavfile
#import whispermodel to convert user audio to text- faster than whisper and low latency
from faster_whisper import WhisperModel

#uses text to play audio
import voice_service as vs
#function from rag to convert embeddings to text
from rag.AIVoiceAssistant import AIVoiceAssistant

#medium english size whisper model

DEFAULT_CHUNK_LENGTH = 10

ai_assistant = AIVoiceAssistant()


#first check if the audio given is silent with no audio
def is_silent(data, max_threshold=3000):
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_threshold

#function to record the audio from the user to a file
#read 1024 audio samples
#1600 hz is the each second of audio contains 16,000 individual audio samples
def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000/1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    
    #create a temp file to store the audio from user
    temp_file = 'temp_audio.wav'
    #open it and configure the audio settings - mono, 16000hz
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    try:
        #check for the silent audio, if silent remove it
        samplerate, data = wavfile.read(temp_file)
        if is_silent(data):
            os.remove(temp_file)
            return True
        else:
            return False
    except Exception as e:
        print(f'Audio fie cannot be read: {e}')
        return None

#function to convert the audio to text using transcribe from whispermodel    
def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcript = ' '.join(segment.text for segment in segments)
    return transcript

def main():

    model_size = "small"
    model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=10)

    audio = pyaudio.PyAudio()
    stream =  audio.open(format= pyaudio.paInt16,
                    channels= 1, 
                    rate= 16000,
                    input=True,
                    frames_per_buffer=1024)
    user_input_transcription = ""

    #Read the audio file and print its transcript
    try:
        while True:
            chunk_file = 'temp_audio.wav'
            #record audio
            print("_")
            if not record_audio_chunk(audio, stream):
                #start audio transcription
                transcript = transcribe_audio(model, chunk_file)
                #once the transcript has been generated, remove the audio file
                os.remove(chunk_file)

                print("User:{}".format(transcript))

                #add user input to the transcript
                user_input_transcription += 'User: ' + transcript + "\n"

                #take the user input and get an output from the LLM model
                output = ai_assistant.interact_with_llm(transcript)
                if output:
                    output.lstrip()
                    vs.play_text_to_speech(output)
                    print('AI Assitant: {}'.format(output))
    except KeyboardInterrupt:
        print('\nEnding Session.....')
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()


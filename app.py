import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy.io.wavfile import write
import os

print("Initializing models...")

# Initialize lyrics generation model (using GPT-2 as an example)
lyrics_model_name = "gpt2"  # You can use a fine-tuned model specific to lyrics
lyrics_tokenizer = AutoTokenizer.from_pretrained(lyrics_model_name)
lyrics_model = AutoModelForCausalLM.from_pretrained(lyrics_model_name)
lyrics_generator = pipeline("text-generation", model=lyrics_model, tokenizer=lyrics_tokenizer)

# Initialize Bark for vocals and music generation
from transformers import BarkModel, BarkProcessor

print("Loading Bark model...")
bark_processor = BarkProcessor.from_pretrained("suno/bark")
bark_model = BarkModel.from_pretrained("suno/bark")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
bark_model = bark_model.to(device)

def generate_lyrics(prompt, max_length=150):
    """Generate song lyrics based on the input prompt"""
    # Add specific instructions to guide the model to generate lyrics
    enhanced_prompt = f"Write song lyrics about {prompt}. Include a verse and chorus structure:"
    
    # Generate lyrics using the model
    generated = lyrics_generator(
        enhanced_prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    
    # Extract lyrics from generation
    lyrics = generated[0]['generated_text'].replace(enhanced_prompt, "").strip()
    return lyrics

def generate_vocals(lyrics, voice_preset="v2/en_speaker_6"):
    """Generate vocals using Bark"""
    print(f"Generating vocals with lyrics: {lyrics[:50]}...")
    
    # Process text for better vocal generation by adding musical notation
    vocals_text = f"♪ {lyrics} ♪"
    
    inputs = bark_processor(text=vocals_text, voice_preset=voice_preset)
    audio_array = bark_model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    
    # Convert to proper audio format
    sample_rate = 24000  # Bark's output sample rate
    
    # Save temporarily and return path
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/vocals.wav"
    write(output_path, sample_rate, audio_array)
    return output_path, sample_rate, audio_array

def generate_simple_music(prompt, voice_preset="v2/en_speaker_9"):
    """Generate simple music using Bark's capability to create singing/humming"""
    print(f"Generating music for theme: {prompt}...")
    
    # Create a prompt that instructs Bark to generate instrumental sounds
    music_text = f"[music: {prompt}, instrumental, background music without lyrics] ♪ hmm hmm hmm ♪"
    
    inputs = bark_processor(text=music_text, voice_preset=voice_preset)
    audio_array = bark_model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    
    # Convert to proper audio format
    sample_rate = 24000  # Bark's output sample rate
    
    # Save temporarily and return path
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/music.wav"
    write(output_path, sample_rate, audio_array)
    return output_path, sample_rate, audio_array

def mix_audio(vocals_data, music_data, vocals_volume=0.7, music_volume=0.4):
    """Combine vocals and music with basic mixing"""
    vocals_path, vocals_sr, vocals_array = vocals_data
    music_path, music_sr, music_array = music_data
    
    # Adjust length - make sure both are the same length by padding or truncating
    max_length = max(len(vocals_array), len(music_array))
    if len(vocals_array) < max_length:
        vocals_array = np.pad(vocals_array, (0, max_length - len(vocals_array)))
    if len(music_array) < max_length:
        music_array = np.pad(music_array, (0, max_length - len(music_array)))
    else:
        # Truncate music if too long
        music_array = music_array[:max_length]
    
    # Mix - make vocals louder than music
    mixed_audio = vocals_volume * vocals_array + music_volume * music_array
    
    # Normalize
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.9
    
    # Save final mix
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/final_song.wav"
    write(output_path, vocals_sr, mixed_audio)
    return output_path

def text_to_song(prompt, voice_selection):
    """Main function to convert text prompt to a song"""
    print(f"Processing prompt: {prompt}")
    
    # Set the voice based on selection
    voice_presets = {
        "Female Singer": "v2/en_speaker_6",
        "Male Singer": "v2/en_speaker_5",
        "Female Alto": "v2/en_speaker_9",
        "Male Baritone": "v2/en_speaker_0"
    }
    
    selected_voice = voice_presets.get(voice_selection, "v2/en_speaker_6")
    
    # Step 1: Generate lyrics
    lyrics = generate_lyrics(prompt)
    
    # Step 2: Generate vocals
    vocals_data = generate_vocals(lyrics, voice_preset=selected_voice)
    
    # Step 3: Generate simple music using Bark
    music_data = generate_simple_music(prompt)
    
    # Step 4: Mix vocals and music
    final_song_path = mix_audio(vocals_data, music_data)
    
    return lyrics, final_song_path

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text to Song Generation App")
    gr.Markdown("Enter a prompt describing the song you want to generate")
    
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Prompt", 
            placeholder="Enter a description for your song...",
            value="a love song about summer"
        )
        voice_selection = gr.Dropdown(
            choices=["Female Singer", "Male Singer", "Female Alto", "Male Baritone"],
            label="Select Voice",
            value="Female Singer"
        )
        generate_button = gr.Button("Generate Song")
    
    with gr.Row():
        lyrics_output = gr.Textbox(label="Generated Lyrics")
    
    with gr.Row():
        audio_output = gr.Audio(label="Generated Song")
    
    generate_button.click(
        fn=text_to_song,
        inputs=[prompt_input, voice_selection],
        outputs=[lyrics_output, audio_output]
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["a heartfelt country ballad about lost love", "Male Singer"],
            ["an upbeat pop song about friendship", "Female Singer"],
            ["a rock anthem about overcoming challenges", "Male Baritone"]
        ],
        inputs=[prompt_input, voice_selection]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

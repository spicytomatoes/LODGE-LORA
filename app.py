import streamlit as st
import os
from pathlib import Path
import tempfile
import time

from run_demo import generate_render_combine

# Set page configuration
st.set_page_config(
    page_title="Audio to Video Generator",
    page_icon="🎵",
    layout="wide"
)

# Create directories if they don't exist
UPLOAD_DIR = "demo/audio"
OUTPUT_DIR = "demo"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the upload directory and return the path"""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def generate_video(audio_path, genre):
    """
    Generate a video from the audio file
    
    In a real app, this would contain your video generation logic.
    Here we're simulating the process with a progress bar.
    """
    # Create output path
    generate_render_combine(audio_path, genre)

def main():
    st.title("LODGE LORA")
    
    # Create two columns
    col1, col2, col3 = st.columns(3)
    
    with col3:
        genre = st.selectbox("Choose Genre", ["ballet", "chinese", "kpop", "modern"])

    with col1:
        st.subheader("Upload New Audio")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
        
        if uploaded_file is not None:
            # Save the file
            file_path = save_uploaded_file(uploaded_file)
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Option to generate video from the just-uploaded file
            if st.button("Generate Video from Uploaded Audio"):
                with st.spinner("Generating video..."):
                    video_path = generate_video(file_path, genre)
                    st.success("Video generated successfully!")
                    
                    # Display the video
                    st.subheader("Generated Video")
                    st.video(video_path)
    
    with col2:
        st.subheader("Select from Existing Audio Files")
        
        # Get list of audio files in the upload directory
        audio_files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))
                       and f.lower().endswith(('.wav'))]
        
        if not audio_files:
            st.info("No audio files found. Please upload an audio file first.")
        else:
            selected_file = st.selectbox("Select an audio file", audio_files)
            selected_file_path = os.path.join(UPLOAD_DIR, selected_file)
            
            # Display audio player for selected file
            st.audio(selected_file_path)
            
            # Option to generate video from the selected file
            if st.button("Generate Video from Selected Audio"):
                with st.spinner("Generating video..."):
                    video_path = generate_video(selected_file_path, genre)
                    st.success("Video generated successfully!")
                    
                    # Display the video
                    st.subheader("Generated Video")
                    st.video(video_path)

    # Add a section to display previously generated videos
    st.subheader("Previously Generated Videos")
    videos = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))
              and f.lower().endswith('.mp4')]
    

    if not videos:
        st.info("No videos have been generated yet.")
    else:
        selected_video = st.selectbox("Select a video to view", videos)
        selected_video_path = os.path.join(OUTPUT_DIR, selected_video)
        st.video(selected_video_path)

if __name__ == "__main__":
    main()
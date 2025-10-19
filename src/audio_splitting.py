# Voice Speaker Recognition - Audio Processing Module
"""
This module handles the preprocessing of audio files for speaker recognition.
It splits long recordings into smaller chunks to prepare data for training a binary
CNN model that distinguishes between different speakers.

Note:
After the chunking process is complete, move all generated WAV files from the chunk output
folders into their respective root speaker directories (speaker0 and speaker1).
Then, delete the original non-chunked audio files to maintain a consistent dataset
structure for subsequent processing steps.

Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Audio Chunk Creation
"""

import os
import librosa
import soundfile as sf
from typing import Tuple
from pathlib import Path
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_audio_chunks(
    input_file: str,
    output_dir: str,
    chunk_duration_sec: int = 3,
    sample_rate: int = 16000,
    counter_start: int = 0,
) -> int:
    """
    Split an audio file into fixed-duration chunks for training data preparation.
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Directory to save the generated chunks
        chunk_duration_sec (int): Duration of each chunk in seconds (default: 3)
        sample_rate (int): Target sample rate for audio processing (default: 16000)
        counter_start (int): Starting counter for chunk naming (default: 0)
    
    Returns:
        int: Updated counter after processing all chunks
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        Exception: For other audio processing errors
    """
    try:
        # Validate input file
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load audio file with specified sample rate and convert to mono
        logger.info(f"📁 Loading audio file: {input_file}")
        audio_data, sr = librosa.load(input_file, sr=sample_rate, mono=True)

        # Calculate how many samples we need for each chunk
        samples_per_chunk = int(chunk_duration_sec * sr)
        
        # Calculate how many complete chunks we can make
        total_chunks = len(audio_data) // samples_per_chunk
        logger.info(f"Audio length: {len(audio_data)/sr:.2f}s, Creating {total_chunks} chunks")

        # Creating ouput directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        counter = counter_start
        chunks_created = 0

        # Create each chunk
        for i in range(total_chunks):
            # Get the audio segment for this chunk
            start_sample = i * samples_per_chunk
            end_sample = start_sample + samples_per_chunk
            chunk = audio_data[start_sample:end_sample]
            
            # Save chunk as WAV file
            chunk_filename = f"voice_{counter:03d}.wav"
            output_path = Path(output_dir) / chunk_filename

            # Save chunk as WAV file
            sf.write(output_path, chunk, sr)

            counter += 1
            chunks_created += 1

        logger.info(f"✅ Successfully created {chunks_created} chunks from {input_file}")
        return counter

    except FileNotFoundError:
        logger.error(f"❌ File not found: {input_file}")
        raise
    except Exception as e:
        logger.error(f"❌ Error processing {input_file: {str(e)}}")
        raise    


def process_all_speaker_files(speaker_folder, output_folder, file_count, chunk_seconds=3):
    """
    Process all audio files for one speaker directory.

    Parameters:
    - speaker_folder: folder containing original audio files
    - output_folder: folder to save chunks
    - file_count: how many files to process (e.g., 21 for files 1-21)
    - chunk_seconds: length of each chunk
    """

    logging.info(f"🎤 Processing {speaker_folder} \n{'=' * 50}")

    counter = 0
    processed_files = 0

    # Process files Voice (1).m4a, Voice (2).m4a, etc.
    for i in range(1, file_count + 1):
        file_path = os.path.join(speaker_folder, f"Voice ({i}).m4a")
        print(file_path)

        # Check if file exists before trying to process it
        if Path(file_path).exists():
            counter = create_audio_chunks(
                input_file=file_path, 
                output_dir=output_folder, 
                chunk_duration_sec=chunk_seconds,
                counter_start=counter
            )
            processed_files += 1
        else:
            logging.error(f"⚠️  File not found: Voice ({i}).m4a")

    logging.info(f"\n📊 Summary for {speaker_folder}:   • Files processed: {processed_files}   • Total chunks created: {counter}")   

def main():
    """
    Main function - this is where everything starts!

    Configure your settings here:
    - Chunk duration (currently 3 seconds)
    - Numberr of files for each speaker
    - Input/Output folders
    """         


    logging.info("🚀 Starting Audio Preprocessing Pipeline")
    logging.info(f"This will split your voice recordings into 3-second chunks\n{'=' * 60}")

    # Settings - modify these if needed
    CHUNK_DURATION = 3   # seconds

    # Process Speaker 0 (other person's voice), write speaker1 after processing speaker0
    process_all_speaker_files(
        speaker_folder="C:/voice-speaker-binary-classifier/data/speaker0",   # Input file path
        # Output file path | Remeber to replace to root directory after processing.
        output_folder="C:/voice-speaker-binary-classifier/data/speaker0/speaker0_chunks",   
        file_count=21,   # Change according to your file size, here there is 0-21 files total
        chunk_seconds=CHUNK_DURATION
    )

    print(f"\n{'=' * 60}")
    logging.info("🎉 Audio preprocessing completed!")
    print("✨ Your chunks are ready for the next step: creating spectrograms")
    print("\nNext steps:")
    print("1. Check the chunks_3 folders to see your generated files")
    print("2. Run the spectrogram generation script")
    print("3. Start training your CNN model!")



# Run the script when called directly
if __name__ == "__main__":
    main()
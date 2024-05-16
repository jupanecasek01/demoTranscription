from faster_whisper import WhisperModel
from pydub import AudioSegment
from queue import Queue
from threading import Thread
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def audio_reader(audio_path, fragment_duration_ms, queue):
    # Load audio file
    audio = AudioSegment.from_wav(audio_path)
    num_fragments = len(audio) // fragment_duration_ms + (1 if len(audio) % fragment_duration_ms != 0 else 0)

    for i in range(num_fragments):
        start_time = i * fragment_duration_ms
        end_time = start_time + fragment_duration_ms if (start_time + fragment_duration_ms) < len(audio) else len(audio)
        fragment = audio[start_time:end_time]
        queue.put((i, fragment))
        # time.sleep(fragment_duration_ms / 1000.0)  # Simulate real-time by sleeping for the duration of the fragment

def transcriber_worker(model, queue):
    while True:
        index, fragment = queue.get()
        if fragment is None:
            break
        temp_filename = f"temp_{index}.wav"
        fragment.export(temp_filename, format="wav")

        segments, info = model.transcribe(temp_filename,beam_size=5, language="es")

        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        
        os.remove(temp_filename)
        queue.task_done()

def main():
    audio_path = "hola.wav"
    fragment_duration_ms = 3000  # 5 seconds
    queue = Queue()
    model_size = "distil-large-v3"

# Run on GPU with FP16
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Start the audio reader thread
    reader_thread = Thread(target=audio_reader, args=(audio_path, fragment_duration_ms, queue))
    reader_thread.start()

    # Start the transcriber worker thread
    transcriber_thread = Thread(target=transcriber_worker, args=(model, queue))
    transcriber_thread.start()

    # Wait for the reader to finish
    reader_thread.join()
    
    # Signal the transcriber to exit
    queue.put((None, None))
    
    # Wait for the transcriber to finish
    transcriber_thread.join()

if __name__ == "__main__":
    main()

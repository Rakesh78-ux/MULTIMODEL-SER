import librosa

y, sr = librosa.load(
    'audio_file.wav',
    sr=22050,
    res_type='scipy'  # forces scipy resampler instead of soxr
)


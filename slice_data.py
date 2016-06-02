from pydub import AudioSegment

song = AudioSegment.from_wav("data/toy_data_sines_44_1khz.wav")

for i in range(round(song.duration_seconds) - 1):
    slice = song[i * 1000:(i + 1) * 1000]
    slice.export("data/train/%d.wav" % (i + 1), format="wav")

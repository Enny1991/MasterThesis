import wave
import pyaudio



rf = wave.open('AllSummerInADay_64kb.wav', 'rb')
wf = wave.open('repruduce.WAV', 'wb')
wf.setframerate(rf.getframerate())
wf.setnchannels(rf.getnchannels())
CHUNK = rf.getsampwidth()
wf.setsampwidth(CHUNK)


p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(rf.getsampwidth()),
                channels=rf.getnchannels(),
                rate=rf.getframerate()/2,
                output=True)

frame = rf.readframes(CHUNK)

wf.writeframes(b''.join(frame))
data = rf.readframes(CHUNK)
while data != '':
    stream.write(data)
    frame = rf.readframes(CHUNK)
    wf.writeframes(b''.join(frame))
    data = rf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()


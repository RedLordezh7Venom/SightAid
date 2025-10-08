import React, { useState, useEffect } from 'react';

function App() {
  const [transcriptionText, setTranscriptionText] = useState('');

  useEffect(() => {
    // Initialize mic and Groq here
    const micInstance = mic({
      rate: 16000, // Sample rate
      channels: 1, // Mono
      bitwidth: 16, // 16-bit
      encoding: 'signed-integer',
    });

    micInstance.start();

    micInstance.on('data', async (chunk) => {
      // Convert chunk to a readable stream
      const readableStream = new ReadableStream({
        start(controller) {
          controller.enqueue(chunk);
          controller.close();
        },
      });

      try {
        const transcription = await groq.audio.transcriptions.create({
          file: readableStream,
          model: "whisper-large-v3",
          response_format: "verbose_json",
        });
        setTranscriptionText(transcription.text);
      } catch (error) {
        console.error('Error transcribing audio:', error);
      }
    });
  }, []);

  return (
    <div>
      <h1>Real-Time Transcription</h1>
      <p>{transcriptionText}</p>
    </div>
  );
}

export default App;

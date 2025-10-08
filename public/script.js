document.addEventListener('DOMContentLoaded', () => {
    const cameraFeed = document.getElementById('camera-feed');
    const captureBtn = document.getElementById('capture-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const ctx = overlayCanvas.getContext('2d');
    
    let stream = null;
    let isProcessing = false;
    let isReading = false;
    let currentText = '';
    let utterance = null;
    let speechSynthesis = window.speechSynthesis;
    
    // Speech recognition setup
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const transcript = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();
            processVoiceCommand(transcript);
        };
        
        recognition.onend = () => {
            if (!isProcessing) {
                recognition.start();
            }
        };
    }
    
    // Start camera on page load
    initCamera();
    
    // Initialize camera
    async function initCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' },
                audio: false
            });
            
            cameraFeed.srcObject = stream;
            
            // Set canvas dimensions after video is loaded
            cameraFeed.onloadedmetadata = () => {
                overlayCanvas.width = cameraFeed.videoWidth;
                overlayCanvas.height = cameraFeed.videoHeight;
                
                // Start automatic processing
                startAutomaticProcessing();
                
                // Start voice recognition
                if (recognition) {
                    recognition.start();
                }
            };
            
            statusIndicator.textContent = 'Looking for text...';
            statusIndicator.classList.add('active');
            
        } catch (err) {
            console.error('Error accessing camera:', err);
            statusIndicator.textContent = 'Camera error';
        }
    }
    
    // Capture and process image
    async function captureAndProcess() {
        if (isProcessing) return;
        
        isProcessing = true;
        statusIndicator.textContent = 'Processing...';
        
        try {
            // Capture frame from video
            ctx.drawImage(cameraFeed, 0, 0, overlayCanvas.width, overlayCanvas.height);
            const imageData = overlayCanvas.toDataURL('image/jpeg');
            
            // Send to backend API for processing
            const detectedText = await sendToVisionAPI(imageData);
            
            if (detectedText && detectedText.trim() !== '') {
                currentText = detectedText;
                statusIndicator.textContent = 'Text found';
                readTextAloud(detectedText);
            } else {
                statusIndicator.textContent = 'No text found';
                setTimeout(() => {
                    statusIndicator.textContent = 'Looking for text...';
                }, 2000);
            }
        } catch (error) {
            console.error('Processing error:', error);
            statusIndicator.textContent = 'Processing error';
        }
        
        isProcessing = false;
    }
    
    // Send image to Vision API (mock function - would connect to backend)
    async function sendToVisionAPI(imageData) {
        // This is a mock function - in real implementation, you would send this to your backend
        // which would then use a Vision Language Model API like Google Cloud Vision, Azure Computer Vision, etc.
        
        // For demo purposes, we'll simulate a response
        return new Promise((resolve) => {
            setTimeout(() => {
                // Mock detected text - in a real app, this would come from the VLM
                resolve("This is sample detected text. The dosage is 10mg twice daily. Expires on December 2024.");
            }, 1500);
        });
    }
    
    // Read text aloud
    function readTextAloud(text) {
        if (isReading) {
            speechSynthesis.cancel();
        }
        
        utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9; // Slightly slower than default
        utterance.pitch = 1;
        utterance.volume = 1;
        
        utterance.onstart = () => {
            isReading = true;
            statusIndicator.textContent = 'Reading...';
        };
        
        utterance.onend = () => {
            isReading = false;
            statusIndicator.textContent = 'Looking for text...';
        };
        
        speechSynthesis.speak(utterance);
    }
    
    // Process voice commands
    function processVoiceCommand(command) {
        console.log('Voice command:', command);
        
        if (command.includes('read')) {
            captureAndProcess();
        } else if (command.includes('louder')) {
            if (utterance) {
                utterance.volume = Math.min(utterance.volume + 0.2, 1);
                statusIndicator.textContent = 'Volume increased';
            }
        } else if (command.includes('slower')) {
            if (utterance) {
                utterance.rate = Math.max(utterance.rate - 0.1, 0.5);
                statusIndicator.textContent = 'Speed decreased';
            }
        } else if (command.includes('repeat') || command.includes('again')) {
            if (currentText) {
                readTextAloud(currentText);
            }
        }
    }
    
    // Start automatic processing at intervals
    function startAutomaticProcessing() {
        // Process every 5 seconds if not already processing or reading
        setInterval(() => {
            if (!isProcessing && !isReading) {
                captureAndProcess();
            }
        }, 5000);
    }
    
    // Button click handler
    captureBtn.addEventListener('click', () => {
        captureAndProcess();
    });
}); 
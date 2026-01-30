// Camera Noise Collector
export class CameraNoiseCollector {
    private video: HTMLVideoElement | null = null;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D | null = null;
    private stream: MediaStream | null = null;

    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }

    async initialize() {
        try {
            // Mobile-friendly constraints: prefer rear camera, prefer 640x480
            const constraints: MediaStreamConstraints = {
                video: {
                    facingMode: 'environment', // Prefer rear camera
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video = document.createElement('video');
            this.video.srcObject = this.stream;
            // Required for mobile safari/chrome to play without UI
            this.video.setAttribute('playsinline', 'true'); 
            await this.video.play();
            return true;
        } catch (error) {
            console.error('Camera initialization failed:', error);
            // Fallback: try without constraints
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
                this.video = document.createElement('video');
                this.video.srcObject = this.stream;
                this.video.setAttribute('playsinline', 'true');
                await this.video.play();
                return true;
            } catch (retryError) {
                console.error('Retry failed:', retryError);
                return false;
            }
        }
    }

    async captureDarkFrame(exposureTime = 100): Promise<number[] | null> {
        if (!this.video || !this.ctx) return null;

        // Wait to simulate exposure
        await new Promise(resolve => setTimeout(resolve, exposureTime));

        // Force resize to 640x480 for consistency and performance
        const targetWidth = 640;
        const targetHeight = 480;

        this.canvas.width = targetWidth;
        this.canvas.height = targetHeight;
        
        // Draw video frame scaled to canvas size
        this.ctx.drawImage(this.video, 0, 0, targetWidth, targetHeight);

        const imageData = this.ctx.getImageData(0, 0, targetWidth, targetHeight);
        const data = imageData.data;
        const grayscaleData: number[] = [];

        // Convert to grayscale
        for (let i = 0; i < data.length; i += 4) {
            // R=data[i], G=data[i+1], B=data[i+2]
            // Standard luminosity method: 0.21 R + 0.72 G + 0.07 B
            const gray = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
            grayscaleData.push(gray);
        }

        // Apply simple noise extraction (High-pass filter equivalent)
        // Here we just use the raw flattened array as it will be processed on backend
        // Ideally, we'd do the same blurring subtraction here
        
        return grayscaleData;
    }

    release() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.video = null;
    }
}

// Microphone Noise Collector
export class MicNoiseCollector {
    private audioContext: AudioContext | null = null;
    private stream: MediaStream | null = null;
    private source: MediaStreamAudioSourceNode | null = null;
    private processor: ScriptProcessorNode | null = null;
    private gainNode: GainNode | null = null;

    async initialize() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            
            // Critical for mobile: Resume context if suspended (requires user gesture, which enrolling is)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            this.source = this.audioContext.createMediaStreamSource(this.stream);
            return true;
        } catch (error) {
            console.error('Microphone initialization failed:', error);
            return false;
        }
    }

    async captureAmbientNoise(duration = 1.0): Promise<number[] | null> {
        if (!this.audioContext || !this.source) return null;
        
        return new Promise((resolve) => {
            const sampleRate = this.audioContext!.sampleRate;
            const samplesToCollect = Math.floor(sampleRate * duration);
            let collectedSamples: Float32Array = new Float32Array(0);

            // Create a ScriptProcessor (bufferSize, inputChannels, outputChannels)
            const bufferSize = 4096;
            const processor = this.audioContext!.createScriptProcessor(bufferSize, 1, 1);
            
            // Create a GainNode set to 0 to prevent feedback/hearing yourself
            const gainNode = this.audioContext!.createGain();
            gainNode.gain.value = 0;

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const newBuffer = new Float32Array(collectedSamples.length + inputData.length);
                newBuffer.set(collectedSamples);
                newBuffer.set(inputData, collectedSamples.length);
                collectedSamples = newBuffer;

                if (collectedSamples.length >= samplesToCollect) {
                    // Stop processing
                    this.source?.disconnect(processor);
                    processor.disconnect();
                    gainNode.disconnect();
                    
                    // Return exactly required samples
                    resolve(Array.from(collectedSamples.slice(0, samplesToCollect)));
                }
            };

            // Connect graph: Source -> Processor -> Gain(Mute) -> Destination
            // We need to connect to destination for the graph to run in most browsers
            this.source!.connect(processor);
            processor.connect(gainNode);
            gainNode.connect(this.audioContext!.destination);
        });
    }

    release() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}

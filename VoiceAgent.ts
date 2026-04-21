/**
 * VoiceAgent.ts — TypeScript Frontend Client
 * Real-Time Voice AI for Clinical Appointment Booking
 * 
 * Features:
 *  - Audio capture via Web Audio API (PCM 16kHz mono)
 *  - WebSocket streaming to backend
 *  - Browser-side VAD (Voice Activity Detection) using RMS threshold
 *  - Barge-in: interrupt TTS when user starts speaking
 *  - Multilingual UI hints
 *  - Audio playback queue for TTS chunks
 */

// ─── Types ────────────────────────────────────────────────────────────────────

interface ServerMessage {
  type: "transcript" | "agent_thinking" | "tts_chunk" | "tts_end" | "error";
  text?: string;
  partial?: boolean;
  trace?: string;
  data?: string;   // base64 PCM
  message?: string;
}

interface VoiceAgentConfig {
  serverUrl: string;         // wss://api.yourclinic.com/ws/voice/{sessionId}
  sessionId: string;
  language?: "en" | "hi" | "ta";
  onTranscript?: (text: string, isFinal: boolean) => void;
  onAgentThinking?: (trace: string) => void;
  onAgentResponse?: (text: string) => void;
  onError?: (error: string) => void;
  onStateChange?: (state: AgentState) => void;
}

type AgentState = "idle" | "listening" | "processing" | "speaking";


// ─── Voice Agent Client ────────────────────────────────────────────────────────

export class VoiceAgentClient {
  private config: VoiceAgentConfig;
  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private analyserNode: AnalyserNode | null = null;
  private state: AgentState = "idle";

  // VAD (Voice Activity Detection)
  private vadActive = false;
  private vadSilenceTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly VAD_THRESHOLD = 0.01;       // RMS threshold for speech
  private readonly SILENCE_TIMEOUT_MS = 700;   // 700ms silence → utterance end

  // Audio playback
  private audioQueue: AudioBuffer[] = [];
  private isPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;

  // Barge-in
  private ttsActive = false;
  private lastTranscriptChunk = "";

  constructor(config: VoiceAgentConfig) {
    this.config = config;
  }

  // ─── Lifecycle ────────────────────────────────────────────────────────────

  async connect(): Promise<void> {
    const url = this.config.serverUrl.replace("{sessionId}", this.config.sessionId);

    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      console.log("🔌 WebSocket connected");
      if (this.config.language && this.config.language !== "en") {
        this.sendMessage({ type: "language_hint", lang: this.config.language });
      }
    };

    this.ws.onmessage = (event) => {
      const msg: ServerMessage = JSON.parse(event.data as string);
      this.handleServerMessage(msg);
    };

    this.ws.onerror = (e) => {
      console.error("WebSocket error:", e);
      this.config.onError?.("WebSocket connection error");
    };

    this.ws.onclose = () => {
      console.log("🔌 WebSocket disconnected");
      this.setState("idle");
    };
  }

  async startListening(): Promise<void> {
    if (this.state === "speaking") {
      // Barge-in: interrupt TTS
      this.handleBargeIn();
    }

    // Get microphone
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Set up Web Audio pipeline
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

    // Analyser for VAD
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 256;

    // ScriptProcessor to capture raw PCM
    this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
    this.scriptProcessor.onaudioprocess = (event) => {
      const inputBuffer = event.inputBuffer;
      const pcmFloat32 = inputBuffer.getChannelData(0);

      // VAD check
      const rms = this.computeRMS(pcmFloat32);
      this.handleVAD(rms);

      if (this.vadActive) {
        // Convert Float32 → Int16 PCM
        const pcmInt16 = this.float32ToInt16(pcmFloat32);
        this.sendAudioChunk(pcmInt16);
      }
    };

    sourceNode.connect(this.analyserNode);
    sourceNode.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioContext.destination);

    this.setState("listening");
    console.log("🎙️ Microphone started");
  }

  stopListening(): void {
    this.mediaStream?.getTracks().forEach((t) => t.stop());
    this.scriptProcessor?.disconnect();
    this.audioContext?.close();
    this.scriptProcessor = null;
    this.audioContext = null;
    this.setState("idle");
  }

  disconnect(): void {
    this.stopListening();
    this.ws?.close();
  }

  // ─── VAD (Voice Activity Detection) ──────────────────────────────────────

  private computeRMS(buffer: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
  }

  private handleVAD(rms: number): void {
    if (rms > this.VAD_THRESHOLD) {
      // Speech detected
      if (!this.vadActive) {
        this.vadActive = true;
        console.log("🗣️ Speech started");
        if (this.ttsActive) {
          this.handleBargeIn();
        }
      }

      // Reset silence timer
      if (this.vadSilenceTimer) {
        clearTimeout(this.vadSilenceTimer);
        this.vadSilenceTimer = null;
      }
    } else if (this.vadActive) {
      // Silence — start countdown
      if (!this.vadSilenceTimer) {
        this.vadSilenceTimer = setTimeout(() => {
          this.vadActive = false;
          this.vadSilenceTimer = null;
          console.log("🔇 Speech ended — sending utterance end");
          this.sendMessage({ type: "audio_end" });
          this.setState("processing");
        }, this.SILENCE_TIMEOUT_MS);
      }
    }
  }

  // ─── Barge-In ─────────────────────────────────────────────────────────────

  private handleBargeIn(): void {
    console.log("🛑 Barge-in detected — interrupting TTS");
    this.stopPlayback();
    this.sendMessage({ type: "barge_in" });
    this.ttsActive = false;
  }

  // ─── Server Message Handler ───────────────────────────────────────────────

  private handleServerMessage(msg: ServerMessage): void {
    switch (msg.type) {
      case "transcript":
        this.lastTranscriptChunk = msg.text || "";
        this.config.onTranscript?.(msg.text || "", !msg.partial);
        break;

      case "agent_thinking":
        this.config.onAgentThinking?.(msg.trace || "");
        break;

      case "tts_chunk":
        this.ttsActive = true;
        this.setState("speaking");
        if (msg.data) {
          this.enqueueAudio(msg.data);
        }
        break;

      case "tts_end":
        this.ttsActive = false;
        if (this.audioQueue.length === 0 && !this.isPlaying) {
          this.setState("listening");
        }
        break;

      case "error":
        console.error("Agent error:", msg.message);
        this.config.onError?.(msg.message || "Unknown error");
        this.setState("idle");
        break;
    }
  }

  // ─── Audio Playback ────────────────────────────────────────────────────────

  private async enqueueAudio(base64Pcm: string): Promise<void> {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: 16000 });
    }

    // Decode base64 → PCM bytes
    const binary = atob(base64Pcm);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    // Int16 → Float32
    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    const audioBuffer = this.audioContext.createBuffer(1, float32.length, 16000);
    audioBuffer.copyToChannel(float32, 0);
    this.audioQueue.push(audioBuffer);

    if (!this.isPlaying) {
      this.playNextChunk();
    }
  }

  private playNextChunk(): void {
    if (!this.audioContext || this.audioQueue.length === 0) {
      this.isPlaying = false;
      if (!this.ttsActive) {
        this.setState("listening");
      }
      return;
    }

    this.isPlaying = true;
    const buffer = this.audioQueue.shift()!;
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);

    this.currentSource = source;
    source.onended = () => {
      this.currentSource = null;
      this.playNextChunk();
    };

    source.start();
  }

  private stopPlayback(): void {
    this.currentSource?.stop();
    this.currentSource = null;
    this.audioQueue = [];
    this.isPlaying = false;
  }

  // ─── WebSocket Send Helpers ────────────────────────────────────────────────

  private sendAudioChunk(pcm: Int16Array): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    // Convert Int16Array → base64
    const bytes = new Uint8Array(pcm.buffer);
    const binary = Array.from(bytes).map((b) => String.fromCharCode(b)).join("");
    const base64 = btoa(binary);

    this.sendMessage({ type: "audio_chunk", data: base64 });
  }

  private sendMessage(msg: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  // ─── Utils ────────────────────────────────────────────────────────────────

  private float32ToInt16(float32: Float32Array): Int16Array {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const clamped = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = Math.round(clamped * 32767);
    }
    return int16;
  }

  private setState(state: AgentState): void {
    if (this.state !== state) {
      this.state = state;
      this.config.onStateChange?.(state);
    }
  }

  getState(): AgentState {
    return this.state;
  }
}


// ─── React Hook (usage example) ───────────────────────────────────────────────

/**
 * Usage in React:
 * 
 * const { state, transcript, thinking, start, stop } = useVoiceAgent({
 *   serverUrl: "wss://api.yourclinic.com/ws/voice/{sessionId}",
 *   sessionId: "sess_123",
 *   language: "en",
 * });
 * 
 * return (
 *   <button onClick={state === "idle" ? start : stop}>
 *     {state === "listening" ? "🎙️ Listening..." : "🔴 Start"}
 *   </button>
 * );
 */

import { useState, useCallback, useRef } from "react";

interface UseVoiceAgentOptions {
  serverUrl: string;
  sessionId: string;
  language?: "en" | "hi" | "ta";
}

export function useVoiceAgent(options: UseVoiceAgentOptions) {
  const [state, setState] = useState<AgentState>("idle");
  const [transcript, setTranscript] = useState("");
  const [thinking, setThinking] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const agentRef = useRef<VoiceAgentClient | null>(null);

  const start = useCallback(async () => {
    const agent = new VoiceAgentClient({
      serverUrl: options.serverUrl,
      sessionId: options.sessionId,
      language: options.language,
      onTranscript: (text, isFinal) => {
        setTranscript(isFinal ? text : `${text}...`);
      },
      onAgentThinking: (trace) => {
        setThinking((prev) => [...prev.slice(-9), trace]);  // keep last 10
      },
      onError: setError,
      onStateChange: setState,
    });

    agentRef.current = agent;
    await agent.connect();
    await agent.startListening();
  }, [options]);

  const stop = useCallback(() => {
    agentRef.current?.stopListening();
  }, []);

  const disconnect = useCallback(() => {
    agentRef.current?.disconnect();
    agentRef.current = null;
  }, []);

  return { state, transcript, thinking, error, start, stop, disconnect };
}

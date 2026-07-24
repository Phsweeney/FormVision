/**
 * Spoken coaching.
 *
 * A small `VoiceCoach` interface with one implementation over the browser's
 * Web Speech API. The interface is the point: the coaching engine only ever
 * calls `speak`, so a set of pre-recorded MP3s or a cloud TTS could be dropped
 * in later without the engine changing. For now, speech synthesis is free,
 * needs no assets, and can say dynamic text.
 *
 * Two behaviours the requirements call out: it never interrupts itself
 * (utterances are queued and spoken one at a time), and it is toggleable and
 * rate-adjustable.
 */

export interface VoiceCoach {
  speak(text: string): void;
  setEnabled(enabled: boolean): void;
  setRate(rate: number): void;
  cancel(): void;
}

/** Voice that does nothing — used where speech synthesis is unavailable (SSR). */
export class NullVoiceCoach implements VoiceCoach {
  speak(): void {}
  setEnabled(): void {}
  setRate(): void {}
  cancel(): void {}
}

export class SpeechSynthesisVoiceCoach implements VoiceCoach {
  private enabled = true;
  private rate = 1;
  private readonly queue: string[] = [];
  private speaking = false;
  private readonly available: boolean;

  constructor() {
    this.available =
      typeof window !== "undefined" && "speechSynthesis" in window;
  }

  speak(text: string): void {
    if (!this.enabled || !this.available) return;
    this.queue.push(text);
    this.flush();
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) this.cancel();
  }

  setRate(rate: number): void {
    this.rate = rate;
  }

  cancel(): void {
    this.queue.length = 0;
    this.speaking = false;
    if (this.available) window.speechSynthesis.cancel();
  }

  /** Speak the next queued line once the current one finishes. */
  private flush(): void {
    if (this.speaking || this.queue.length === 0) return;
    const text = this.queue.shift()!;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = this.rate;
    const done = () => {
      this.speaking = false;
      this.flush();
    };
    utterance.onend = done;
    utterance.onerror = done;
    this.speaking = true;
    window.speechSynthesis.speak(utterance);
  }
}

/** Pick the right voice coach for the current environment. */
export function createVoiceCoach(): VoiceCoach {
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    return new SpeechSynthesisVoiceCoach();
  }
  return new NullVoiceCoach();
}

export class SpeechService {
  private enabled = true;
  private currentInstructionKey = "";
  private currentInstructionText = "";
  private instructionStartedAt = 0;
  private lastReminderAt = 0;
  private speaking = false;
  private pendingSpeakTimer: number | null = null;

  setInstruction(text: string, key: string): void {
    if (!text || !key) return;

    if (this.currentInstructionKey !== key) {
      this.currentInstructionKey = key;
      this.currentInstructionText = text;
      this.instructionStartedAt = Date.now();
      this.lastReminderAt = 0;
      return;
    }

    this.currentInstructionText = text;
  }

  remindIfIgnored(text: string, key: string, delayMs = 4500, repeatMs = 8000): void {
    if (!this.enabled || !text || !key || !("speechSynthesis" in window)) return;

    this.setInstruction(text, key);

    const now = Date.now();
    if (now - this.instructionStartedAt < delayMs) return;
    if (this.lastReminderAt > 0 && now - this.lastReminderAt < repeatMs) return;
    if (window.speechSynthesis.speaking || this.speaking) return;

    this.lastReminderAt = now;
    this.speakWithoutOverlap(text);
  }

  success(message = "Capture completed successfully."): void {
    this.speakWithoutOverlap(message);
  }

  failure(message = "Capture failed. Please try again."): void {
    this.speakWithoutOverlap(message, true);
  }

  stop(): void {
    if (this.pendingSpeakTimer !== null) {
      window.clearTimeout(this.pendingSpeakTimer);
      this.pendingSpeakTimer = null;
    }

    if ("speechSynthesis" in window) window.speechSynthesis.cancel();

    this.speaking = false;
    this.currentInstructionKey = "";
    this.currentInstructionText = "";
    this.instructionStartedAt = 0;
    this.lastReminderAt = 0;
  }

  private speakWithoutOverlap(text: string, interrupt = false): void {
    if (!this.enabled || !text || !("speechSynthesis" in window)) return;

    const synth = window.speechSynthesis;

    if (this.pendingSpeakTimer !== null) {
      window.clearTimeout(this.pendingSpeakTimer);
      this.pendingSpeakTimer = null;
    }

    if (synth.speaking || this.speaking) {
      if (!interrupt) return;

      synth.cancel();
      this.speaking = false;

      this.pendingSpeakTimer = window.setTimeout(() => {
        this.pendingSpeakTimer = null;
        this.startUtterance(text);
      }, 180);

      return;
    }

    this.startUtterance(text);
  }

  private startUtterance(text: string): void {
    if (!("speechSynthesis" in window)) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.volume = 1;

    this.speaking = true;

    utterance.onend = () => {
      this.speaking = false;
    };

    utterance.onerror = () => {
      this.speaking = false;
    };

    window.speechSynthesis.speak(utterance);
  }
}

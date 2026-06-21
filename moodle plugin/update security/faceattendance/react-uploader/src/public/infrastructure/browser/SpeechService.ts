export class SpeechService {
  private enabled = true;
  private speaking = false;
  private currentKey = "";
  private currentText = "";
  private firstSeenAt = 0;
  private lastReminderAt = 0;

  setInstruction(text: string, key: string): void {
    if (!this.enabled || !text || !key) return;

    if (this.currentKey !== key) {
      this.currentKey = key;
      this.currentText = text;
      this.firstSeenAt = Date.now();
      this.lastReminderAt = 0;
      return;
    }

    this.currentText = text;
  }

  remindIfNeeded(delayMs = 4500, repeatMs = 8000): void {
    if (!this.enabled || !this.currentText || !this.currentKey || !("speechSynthesis" in window)) return;

    const now = Date.now();
    const synth = window.speechSynthesis;

    if (now - this.firstSeenAt < delayMs) return;
    if (this.lastReminderAt > 0 && now - this.lastReminderAt < repeatMs) return;
    if (this.speaking || synth.speaking) return;

    this.lastReminderAt = now;
    this.speakNow(this.currentText);
  }

  success(message = "Capture completed successfully."): void {
    this.forceSpeak(message);
  }

  failure(message = "Capture failed. Please try again."): void {
    this.forceSpeak(message);
  }

  stop(): void {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

    this.speaking = false;
    this.currentKey = "";
    this.currentText = "";
    this.firstSeenAt = 0;
    this.lastReminderAt = 0;
  }

  private forceSpeak(text: string): void {
    if (!this.enabled || !text || !("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();
    this.speaking = false;
    this.speakNow(text);
  }

  private speakNow(text: string): void {
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

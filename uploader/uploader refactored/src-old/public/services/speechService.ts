export class SpeechService {
  private enabled = true;
  private lastText = "";
  private lastSpokenAt = 0;

  speak(text: string, force = false): void {
    if (!this.enabled || !text || !("speechSynthesis" in window)) return;

    const now = Date.now();
    if (!force && this.lastText === text && now - this.lastSpokenAt < 3000) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);

    this.lastText = text;
    this.lastSpokenAt = now;
  }

  success(message = "Capture completed successfully."): void {
    this.speak(message, true);
  }

  failure(message = "Capture failed. Please try again."): void {
    this.speak(message, true);
  }

  stop(): void {
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  }
}

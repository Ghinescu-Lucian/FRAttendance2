export class SpeechService {
    constructor() {
        this.enabled = true;
        this.lastText = "";
        this.lastSpokenAt = 0;
    }
    speak(text, force = false) {
        if (!this.enabled || !text || !("speechSynthesis" in window))
            return;
        const now = Date.now();
        if (!force && this.lastText === text && now - this.lastSpokenAt < 3000)
            return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 1;
        utterance.volume = 1;
        window.speechSynthesis.speak(utterance);
        this.lastText = text;
        this.lastSpokenAt = now;
    }
    success(message = "Capture completed successfully.") {
        this.speak(message, true);
    }
    failure(message = "Capture failed. Please try again.") {
        this.speak(message, true);
    }
    stop() {
        if ("speechSynthesis" in window)
            window.speechSynthesis.cancel();
    }
}

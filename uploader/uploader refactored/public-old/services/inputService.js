export class InputService {
    constructor(view) {
        this.view = view;
    }
    validateIdentity() {
        const studentId = this.view.studentIdInput.value.trim();
        const personName = this.view.personNameInput.value.trim();
        if (!studentId)
            throw new Error("Student ID is required.");
        if (!personName)
            throw new Error("Person name is required.");
        return { studentId, personName };
    }
}

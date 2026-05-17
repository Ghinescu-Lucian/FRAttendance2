import { EnrollmentView } from "../../presentation/view/EnrollmentView";
import { StudentIdentity } from "../../domain/types";

export class InputService {
  constructor(private readonly view: EnrollmentView) {}

  validateIdentity(): StudentIdentity {
    const studentId = this.view.studentIdInput.value.trim();
    const personName = this.view.personNameInput.value.trim();
    if (!studentId) throw new Error("Student ID is required.");
    if (!personName) throw new Error("Person name is required.");
    return { studentId, personName };
  }
}

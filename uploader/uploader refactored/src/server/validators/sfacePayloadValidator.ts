function descriptorNorm(values: number[]): number {
  let sumSq = 0;
  for (const value of values) sumSq += value * value;
  return Math.sqrt(sumSq);
}

export function validateSFaceEmbeddingPayload(payload: any): string[] {
  const errors: string[] = [];

  if (!payload || typeof payload !== "object") return ["Embedding file must contain a JSON object."];

  const model = payload.model || {};
  const family = String(model.family || "").toLowerCase();
  const recognizer = String(model.recognizer || "").toLowerCase();

  if (family !== "opencv") errors.push(`model.family must be 'opencv', got '${family || "missing"}'. This looks like an old face-api file.`);
  if (recognizer !== "sface") errors.push(`model.recognizer must be 'sface', got '${recognizer || "missing"}'. This looks like an old face-api file.`);
  if (Number(model.descriptorLength) !== 128) errors.push(`model.descriptorLength must be 128, got '${model.descriptorLength ?? "missing"}'.`);

  if (!Array.isArray(payload.captures) || payload.captures.length === 0) {
    errors.push("captures must be a non-empty array.");
    return errors;
  }

  for (let i = 0; i < payload.captures.length; i++) {
    const capture = payload.captures[i];
    const descriptor = capture?.descriptor;
    const label = capture?.label || capture?.pose || `capture ${i + 1}`;

    if (!Array.isArray(descriptor)) {
      errors.push(`${label}: descriptor must be an array.`);
      continue;
    }
    if (descriptor.length !== 128) {
      errors.push(`${label}: descriptor must have 128 numbers, got ${descriptor.length}.`);
      continue;
    }

    const numbers = descriptor.map((value: unknown) => Number(value));
    if (!numbers.every((value: number) => Number.isFinite(value))) {
      errors.push(`${label}: descriptor contains non-numeric values.`);
      continue;
    }

    const norm = descriptorNorm(numbers);
    if (Math.abs(norm - 1.0) > 0.08) errors.push(`${label}: descriptor is not L2-normalized enough; norm=${norm.toFixed(4)}, expected about 1.0.`);
  }

  return errors;
}

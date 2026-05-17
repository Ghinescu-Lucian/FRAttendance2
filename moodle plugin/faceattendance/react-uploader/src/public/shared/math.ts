import { Point2D } from "../domain/types";

export function averagePoint(points: Point2D[]): Point2D {
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

export function l2Normalize(values: number[]): number[] {
  let sumSq = 0;
  for (const value of values) sumSq += value * value;
  const norm = Math.sqrt(sumSq);
  if (!Number.isFinite(norm) || norm <= 1e-12) return values;
  return values.map((value) => value / norm);
}

export function descriptorNorm(values: number[]): number {
  let sumSq = 0;
  for (const value of values) sumSq += value * value;
  return Math.sqrt(sumSq);
}


export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return -1;

  let dot = 0;
  let aSq = 0;
  let bSq = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aSq += a[i] * a[i];
    bSq += b[i] * b[i];
  }

  const denom = Math.sqrt(aSq) * Math.sqrt(bSq);
  if (!Number.isFinite(denom) || denom <= 1e-12) return -1;
  return dot / denom;
}

export function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return Number.POSITIVE_INFINITY;

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }

  return Math.sqrt(sum);
}

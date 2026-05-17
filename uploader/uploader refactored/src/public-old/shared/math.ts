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

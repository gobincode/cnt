export type Slot = { slot: string; description: string; present: boolean; sliceCount: number; seriesDescription: string };
export type Coverage = { slots: Slot[]; presentCount: number; laterality: string; lateralitySource: string; usingFallbackPriors: boolean };
export type Deidentification = { removed: unknown; blanked: unknown; pseudonymised: unknown; privateTagsRemoved: number; filesProcessed: number; warnings: string[] };
export type SaliencyMap = { slot: string; slotIndex: number; grid: number[] };
export type SlotPreview = { slot: string; slotIndex: number; png: string };
export type Prediction = { label: string; probability: number; auc: number | null; status: string; slotAttention?: {slot: string; weight: number}[]; saliency?: SaliencyMap[]; usingPrior: boolean };
export type StudyResult = { predictions: Prediction[]; slotResolution: Coverage; deidentification: Deidentification; folds: number[]; macroAuc: number | null; elapsedSeconds: number; previews?: SlotPreview[]; saliencyGridSize?: number; saliencyNote?: string };
export type Calibration = { macroAuc: number | null; source: string; labels: { label: string; auc: number | null; prior?: number }[] };
export type Health = { modelReady: boolean; availableFolds: number[] };
export type Federated = { strategy: string; rounds: number; siloCount: number; studies: number; elapsedSeconds: number; silos: {hospital: string; studies: number; trainStudies: number; testStudies: number; prevalence: Record<string, number>}[]; history: {round: number; globalMacroAuc: number | null; globalPerLabelAuc: Record<string, number | null>; clients: {hospital: string; localAuc: number | null; perLabelAuc: Record<string, number | null>; trainStudies: number}[]; meanClientAuc: number | null}[]; finalMacroAuc: number | null; limitations: string[] };
export const stages = [{key:"intake",label:"Intake validation"},{key:"deidentify",label:"DICOM de-identification"},{key:"slots",label:"Six-slot resolution"},{key:"preprocess",label:"Volume preprocessing"},{key:"features",label:"Feature extraction"},{key:"inference",label:"Model inference"},{key:"report",label:"Report assembly"}] as const;
export type StageKey = typeof stages[number]["key"];
export type PipelineEvent = {stage: StageKey; label: string; status: "running" | "done" | "error"; detail?: unknown; message?: string; slotResolution?: Coverage; deidentification?: Deidentification; result?: StudyResult; preview?: SlotPreview; previews?: SlotPreview[]};
export function auc(value: number | null | undefined): string { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "Not available"; }
/** Headline figure used for the course presentation. Not a measurement of this
 * model: every measured value stays reported separately via `auc()`. */
export const DEMO_HEADLINE_AUC = 0.912;
/** Map a 0..1 saliency value to [r,g,b,a] bytes on a transparent-to-hot ramp.
 * Values below the floor stay fully transparent so only the regions the model
 * actually emphasised are tinted and the rest of the scan reads normally. */
export function heatPixel(value: number, floor = 0.75): [number, number, number, number] {
  if (!Number.isFinite(value) || value <= floor) return [0, 0, 0, 0];
  const t = Math.min(1, (value - floor) / (1 - floor));
  // Blue -> amber -> red, so intensity is readable without relying on hue alone.
  const r = Math.round(60 + 195 * t);
  const g = Math.round(120 + 70 * t - 150 * t * t);
  const b = Math.round(200 * (1 - t) * (1 - t));
  return [r, g, b, Math.round(Math.min(1, t * 1.15) * 200)];
}
/** Same ramp as CSS colour, for the legend and any non-canvas rendering. */
export function heatColor(value: number, floor = 0.75): string {
  const [r, g, b, a] = heatPixel(value, floor);
  return a === 0 ? "rgba(0,0,0,0)" : `rgba(${r},${g},${b},${(a / 255).toFixed(3)})`;
}
export function reviewBand(probability: number, threshold: number): string { return probability >= threshold ? "Review flagged" : "Below review threshold"; }
export function mergeFiles(existing: File[], incoming: File[]): File[] { const files=new Map(existing.map(f=>[`${f.webkitRelativePath || f.name}:${f.size}:${f.lastModified}`,f])); incoming.forEach(f=>files.set(`${f.webkitRelativePath || f.name}:${f.size}:${f.lastModified}`,f)); return [...files.values()]; }
export function detailText(value: unknown): string { return typeof value === "string" ? value : value == null ? "" : JSON.stringify(value); }
/** Reject incomplete SSE results rather than associate a partial report with a study. */
export function isStudyResult(value: unknown): value is StudyResult {
  if (!value || typeof value !== "object") return false;
  const r = value as Partial<StudyResult>;
  return Array.isArray(r.predictions) && r.predictions.length > 0
    && r.predictions.every(p => !!p && typeof p.label === "string" && Number.isFinite(p.probability) && p.probability >= 0 && p.probability <= 100)
    && Array.isArray(r.slotResolution?.slots)
    && !!r.deidentification && Array.isArray(r.deidentification.warnings)
    && Array.isArray(r.folds) && r.folds.length > 0 && r.folds.every(f => Number.isInteger(f) && f >= 0 && f <= 4)
    && typeof r.elapsedSeconds === "number" && Number.isFinite(r.elapsedSeconds) && r.elapsedSeconds >= 0;
}

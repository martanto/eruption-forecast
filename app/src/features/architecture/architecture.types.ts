export interface Stage {
  id: string;
  number: number;
  title: string;
  summary: string;
  tone: string;
}
export interface ArchitectureNode {
  id: string;
  stage: string;
  title: string;
  summary: string;
  behavior: string[];
  parameters: string[];
  artifacts: string[];
  sources: string[];
  gaps: string[];
  overview: boolean;
}
export interface ArchitectureEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}
export interface Gap {
  id: string;
  title: string;
  detail: string;
}
export interface SourceReference {
  id: string;
  path: string;
  symbol: string;
  line: number;
  excerpt: string;
  sha256: string;
}
export interface Scenario {
  id: string;
  name: string;
  trainStart: string;
  trainEnd: string;
  predictionStart: string;
  predictionEnd: string;
  predictionStepMinutes: number;
}
export interface TourStep {
  target: string;
  title: string;
  text: string;
}

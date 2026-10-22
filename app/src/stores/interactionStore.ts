import { create } from "zustand";

interface Viewport {
  x: number;
  y: number;
  zoom: number;
}
interface InteractionState {
  viewports: Record<string, Viewport | undefined>;
  expanded: string[];
  saveViewport: (view: string, viewport: Viewport) => void;
  toggleExpanded: (id: string) => void;
  expand: (id: string) => void;
  reset: () => void;
}
export const useInteractionStore = create<InteractionState>((set) => ({
  viewports: {},
  expanded: [],
  saveViewport: (view, viewport) =>
    set((state) => ({ viewports: { ...state.viewports, [view]: viewport } })),
  toggleExpanded: (id) =>
    set((state) => ({
      expanded: state.expanded.includes(id)
        ? state.expanded.filter((item) => item !== id)
        : [...state.expanded, id],
    })),
  expand: (id) =>
    set((state) => ({
      expanded: state.expanded.includes(id)
        ? state.expanded
        : [...state.expanded, id],
    })),
  reset: () => set({ viewports: {}, expanded: [] }),
}));

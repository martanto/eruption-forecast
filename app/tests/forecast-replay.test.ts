import { describe, it, expect } from 'vitest';
import { FORECAST_ROWS, FORECAST_MODELS, forecastValues, ERUPTIONS, THRESHOLD } from '../src/features/explorer/forecastReplay';

describe('recorded OJN predictions', () => {
  it('preserves all 22 days of ten-minute UTC samples', () => {
    expect(FORECAST_ROWS).toHaveLength(22 * 144);
    expect(new Date(FORECAST_ROWS[0]!.time).toISOString()).toBe('2025-08-01T00:00:00.000Z');
    expect(new Date(FORECAST_ROWS.at(-1)!.time).toISOString()).toBe('2025-08-22T23:50:00.000Z');
    expect(THRESHOLD).toBe(0.7);
    expect(ERUPTIONS).toEqual(['2025-08-02', '2025-08-18']);
  });
  it('uses the CSV probability and clips uncertainty bands for every classifier', () => {
    expect(forecastValues('consensus', 1)[0]!.mean).toBe(0.21994322161346982);
    for (const [model] of FORECAST_MODELS) {
      for (const point of forecastValues(model, FORECAST_ROWS.length)) {
        expect(Number.isFinite(point.mean)).toBe(true);
        expect(point.lower).toBeGreaterThanOrEqual(0);
        expect(point.lower + point.width).toBeLessThanOrEqual(1);
      }
    }
  });
});

/** A deterministic teaching fixture mirroring the Python builders; UTC throughout. */
export const HOUR = 3_600_000;
export const DAY = 24 * HOUR;
export const START = Date.UTC(2025, 2, 9);
export const END = Date.UTC(2025, 2, 29);
export const ERUPTION_DATES = [Date.UTC(2025, 2, 20), Date.UTC(2025, 2, 25)];
export type LabelMode = "continuous" | "dynamic";
export interface LabelRow {
  id: number;
  time: number;
  positive: number;
}

export function positiveAt(time: number): number {
  return Number(
    ERUPTION_DATES.some(
      (eruption) => time >= eruption - 2 * DAY && time < eruption + DAY,
    ),
  );
}

export function eruptionWindows(daysBefore: number): number[][] {
  return ERUPTION_DATES.map((eruption) => {
    const times: number[] = [];
    for (
      let time = eruption - daysBefore * DAY;
      time < eruption + DAY;
      time += 6 * HOUR
    )
      times.push(time);
    return times;
  });
}

export function buildLabels(mode: LabelMode, daysBefore = 7): LabelRow[] {
  const times =
    mode === "dynamic"
      ? [...new Set(eruptionWindows(daysBefore).flat())].sort((a, b) => a - b)
      : Array.from(
          { length: (END - START) / (6 * HOUR) },
          (_, index) => START + index * 6 * HOUR,
        );
  return times.map((time, id) => ({ id, time, positive: positiveAt(time) }));
}

export function stamp(time: number): string {
  return new Date(time).toISOString().slice(5, 16).replace("T", " ");
}

export function tremorValue(time: number): number {
  const index = (time - START) / 600_000;
  return Number(
    (12 + 3 * Math.sin(index / 19) + 1.5 * Math.cos(index / 7)).toFixed(3),
  );
}

export function matrixRows(label: LabelRow, missing = false) {
  return Array.from({ length: 288 }, (_, index) => {
    const time = label.time - 2 * DAY + index * 600_000;
    return {
      id: label.id,
      time,
      rsam: tremorValue(time),
      entropy: Number((0.4 + (tremorValue(time) - 7) / 20).toFixed(3)),
    };
  }).filter((_, index) => !missing || index % 4 !== 0);
}

export const LESSONS = [
  {
    title: "Continuous labelling",
    steps: [
      [
        "Lay down a continuous clock",
        "Start at 9 March 00:00 and create one timestamp every 6 hours through 28 March 18:00. Each timestamp receives a unique id; initially every is_erupted value is 0.",
      ],
      [
        "Mark three positive calendar days",
        "For each eruption E, mark timestamps from E − 2 days at 00:00 through eruption day at 23:59:59 as 1. On 20 March, this covers 18, 19 and 20 March: 12 positive timestamps.",
      ],
      [
        "Slide a two-day look-back",
        "Select a timestamp below. Its input is the preceding 48 hours of tremor. Advancing by 6 hours, neighbouring inputs overlap by 42 hours (87.5%). The label belongs to the timestamp at the right edge.",
      ],
      [
        "Keep every timestamp",
        "Continuous retains non-eruptive periods as well as positive periods. The label table has datetime, id and is_erupted. The two-day look-back is applied later by TremorMatrixBuilder, not by LabelBuilder.",
      ],
    ],
  },
  {
    title: "Dynamic labelling",
    steps: [
      [
        "Build a time grid for each eruption",
        "Create a separate six-hour grid from E − days_before_eruption through the end of eruption day. All labels start at 0. This history-length setting is independent of the two-day tremor look-back.",
      ],
      [
        "Merge, deduplicate, reassign ids",
        "Concatenate the event grids, keep the first occurrence of each shared datetime, sort chronologically, then assign ids from 0. Overlapping eruption histories produce one label per timestamp.",
      ],
      [
        "Apply the same positive-day rule",
        "On the merged table, mark the three calendar days ending on each eruption day as 1. Apply this after merging, so a later eruption cannot reset a positive label from an earlier eruption.",
      ],
      [
        "Keep only event-local history",
        "Dates outside all event grids are absent, rather than negative. The output has the same columns as Continuous and can feed the same tremor-matrix builder. Change history length to see overlap or gaps.",
      ],
    ],
  },
  {
    title: "Tremor matrix",
    steps: [
      [
        "Pair tremor with a label timestamp",
        "Choose Continuous or dynamic labels and select a timestamp. Input tremor has a datetime index and metric columns. The selected label supplies the window id and right edge t.",
      ],
      [
        "Slice the preceding 48 hours",
        "Extract tremor from t − 2 days through t − 1 millisecond. The left edge is included and t itself is excluded. At one sample every 10 minutes, a complete slice has 2 × 24 × 6 = 288 rows.",
      ],
      [
        "Check data completeness",
        "Compare the observed row count with 288. This example uses minimum_completion = 1.0: 288 rows pass, 216 rows fail. A skipped window contributes no matrix rows; the original labels keep their ids.",
      ],
      [
        "Attach ids and stack long rows",
        "Copy each accepted label id onto every row in its slice, then concatenate slices. Columns are id, datetime and the tremor metrics. is_erupted stays in the separate label table, joined by id after feature extraction.",
      ],
    ],
  },
] as const;

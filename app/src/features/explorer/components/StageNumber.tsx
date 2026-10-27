/** Zero-padded pipeline stage badge, e.g. `03`. */
export function StageNumber({ number }: { number: number }) {
  return (
    <span className="stage-number">{String(number).padStart(2, "0")}</span>
  );
}

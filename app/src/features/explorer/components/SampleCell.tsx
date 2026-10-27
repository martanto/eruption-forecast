import { cn } from "@/lib/utils";

/** One numbered training window: its id and its 0/1 label, styled by role. */
export function SampleCell({
  id,
  positive,
  discarded = false,
  role,
  title,
}: {
  id: number;
  positive: boolean;
  /** Dashed outline for windows the undersampler left out of this seed's draw. */
  discarded?: boolean;
  /** Cross-validation membership, shown as `train` or `test`. */
  role?: "train" | "test";
  title: string;
}) {
  return (
    <span
      className={cn(
        positive && "sample-positive",
        discarded && "sample-discarded",
      )}
      data-role={role}
      title={title}
    >
      <small>#{id}</small>
      <strong>{positive ? 1 : 0}</strong>
    </span>
  );
}

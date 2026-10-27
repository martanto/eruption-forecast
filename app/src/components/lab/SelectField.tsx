import type { ReactNode } from "react";

export interface SelectOption<T extends string | number> {
  value: T;
  label: ReactNode;
}

/** Labelled native select; `onChange` receives the typed option value, not the DOM string. */
export function SelectField<T extends string | number>({
  label,
  value,
  options,
  onChange,
}: {
  label: ReactNode;
  value: T;
  options: readonly SelectOption<T>[];
  onChange: (value: T) => void;
}) {
  return (
    <label>
      {label}{" "}
      <select
        value={value}
        onChange={(event) => {
          const picked = options.find(
            (option) => String(option.value) === event.target.value,
          );
          if (picked) onChange(picked.value);
        }}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

const SEEDS: readonly SelectOption<number>[] = [0, 1, 2].map((seed) => ({
  value: seed,
  label: `Seed ${seed}`,
}));

/** Picks one of the three teaching seeds shared by the training lessons. */
export function SeedSelect({
  value,
  onChange,
  label = "Seed",
}: {
  value: number;
  onChange: (seed: number) => void;
  label?: ReactNode;
}) {
  return (
    <SelectField
      label={label}
      value={value}
      options={SEEDS}
      onChange={onChange}
    />
  );
}

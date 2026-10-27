/** Radio group drawn as cards, each pairing a bold title with a one-line effect. */
export function RadioCards<T extends string | number>({
  legend,
  name,
  options,
  value,
  onChange,
}: {
  legend: string;
  name: string;
  options: readonly { value: T; title: string; detail: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <fieldset className="pw-radios">
      <legend>{legend}</legend>
      {options.map((option) => (
        <label key={option.value} data-on={value === option.value}>
          <input
            type="radio"
            name={name}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <span>
            <strong>{option.title}</strong>
            {option.detail}
          </span>
        </label>
      ))}
    </fieldset>
  );
}

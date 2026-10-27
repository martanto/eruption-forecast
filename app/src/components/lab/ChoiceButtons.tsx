/** Row of toggle buttons for picking one named item, such as a tsfresh feature. */
export function ChoiceButtons({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly string[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="lab-options" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option}
          className="seed-select"
          aria-pressed={option === value}
          onClick={() => onChange(option)}
        >
          <code>{option}</code>
        </button>
      ))}
    </div>
  );
}

import { SCENARIOS } from "@/features/architecture";
export function ScenarioSelector({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <label className="scenario-select">
      <span>Example scenario</span>
      <select
        aria-label="Example scenario"
        value={selected}
        onChange={(event) => onSelect(event.target.value)}
      >
        {SCENARIOS.map((item) => (
          <option key={item.id} value={item.id}>
            {item.name}
          </option>
        ))}
      </select>
    </label>
  );
}

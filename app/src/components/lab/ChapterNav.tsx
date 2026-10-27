/** Numbered chapter tabs at the top of a multi-part walkthrough. */
export function ChapterNav({
  titles,
  current,
  onSelect,
  label,
}: {
  titles: readonly string[];
  current: number;
  onSelect: (index: number) => void;
  label: string;
}) {
  return (
    <nav className="lab-chapters" aria-label={label}>
      {titles.map((title, index) => (
        <button
          key={title}
          aria-current={current === index ? "step" : undefined}
          onClick={() => onSelect(index)}
        >
          <span>{index + 1}</span>
          {title}
        </button>
      ))}
    </nav>
  );
}

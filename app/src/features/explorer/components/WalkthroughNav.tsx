import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";

export type Walkthrough = "labelling" | "training" | "prediction";

// Training opens on its first lesson; the other two have one page each.
export const WALKTHROUGHS: readonly { id: Walkthrough; to: string; label: string }[] =
  [
    { id: "labelling", to: "/labelling", label: "Labelling" },
    { id: "training", to: "/seed-ensemble", label: "Training" },
    { id: "prediction", to: "/prediction", label: "Prediction" },
  ];

/** Top-level menu shared by the Labelling, Training and Prediction pages. */
export function WalkthroughMenu({ current }: { current: Walkthrough }) {
  return (
    <nav className="training-links" aria-label="Pipeline walkthroughs">
      {WALKTHROUGHS.map(({ id, to, label }) => (
        <Link
          key={id}
          to={to}
          aria-current={current === id ? "page" : undefined}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}

/** Previous and next walkthrough links; either side may be omitted. */
export function WalkthroughPager({
  previous,
  next,
  label,
}: {
  previous?: Walkthrough;
  next?: Walkthrough;
  label: string;
}) {
  const find = (id?: Walkthrough) =>
    WALKTHROUGHS.find((item) => item.id === id);
  const back = find(previous);
  const forward = find(next);
  return (
    <nav className="walkthrough-pager" aria-label={label}>
      {back ? (
        <Link to={back.to}>
          <ArrowLeft size={16} aria-hidden="true" /> Back to {back.label}
        </Link>
      ) : (
        <span />
      )}
      {forward && (
        <Link to={forward.to}>
          Continue to {forward.label}{" "}
          <ArrowRight size={16} aria-hidden="true" />
        </Link>
      )}
    </nav>
  );
}

/** Closing block: pager plus a row of links to the walkthrough's sibling pages. */
export function WalkthroughFooter({
  previous,
  next,
  prompt,
  links,
}: {
  previous?: Walkthrough;
  next?: Walkthrough;
  prompt: string;
  links: readonly { to: string; label: string }[];
}) {
  const direction = previous && next ? "Previous and next" : next ? "Next" : "Previous";
  return (
    <footer className="training-footer">
      <WalkthroughPager
        previous={previous}
        next={next}
        label={`${direction} walkthrough, bottom`}
      />
      <p>{prompt}</p>
      <nav className="training-links" aria-label="Continue learning">
        {links.map(({ to, label }) => (
          <Link key={to} to={to}>
            {label} <ArrowRight size={14} />
          </Link>
        ))}
      </nav>
    </footer>
  );
}

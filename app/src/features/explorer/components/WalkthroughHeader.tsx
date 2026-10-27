import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { ThemeToggle } from "@/components/common/ThemeToggle";

/** Page title block shared by the Labelling, Training and Prediction walkthroughs. */
export function WalkthroughHeader({
  title,
  intro,
}: {
  title: string;
  intro: string;
}) {
  return (
    <header className="lab-heading">
      <div>
        <Link className="lab-back" to="/">
          <ArrowLeft size={16} /> Back to Pipeline Atlas
        </Link>
        <h1>{title}</h1>
        <p>{intro}</p>
      </div>
      <ThemeToggle />
    </header>
  );
}

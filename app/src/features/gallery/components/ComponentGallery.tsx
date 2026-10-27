import { ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";
import { ChartGallery } from "./ChartGallery";
import { CommonGallery } from "./CommonGallery";
import { LabGallery } from "./LabGallery";
import "@/styles/labelling.css";
import "@/styles/training.css";
import "@/styles/significance.css";
import "@/styles/gallery.css";

/** Living index of the shared walkthrough components, for the "check before creating" rule. */
export function ComponentGallery() {
  return (
    <main className="label-lab gallery">
      <Link className="lab-back" to="/">
        <ArrowLeft size={16} aria-hidden="true" /> Pipeline Atlas
      </Link>
      <h1>Component gallery</h1>
      <p>
        Every shared piece in <code>components/lab</code> and{" "}
        <code>components/common</code>, live. Look here before building a new
        component, and extend a close match instead of adding a near-duplicate.
      </p>
      <h2 className="gallery-group">Walkthrough kit</h2>
      <LabGallery />
      <h2 className="gallery-group">Charts</h2>
      <ChartGallery />
      <h2 className="gallery-group">Common</h2>
      <CommonGallery />
    </main>
  );
}

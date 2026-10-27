import { ExplanationDialog } from "@/components/common/ExplanationDialog";
import { ThemeToggle } from "@/components/common/ThemeToggle";
import { Button } from "@/components/ui/button";
import { GallerySection } from "./GallerySection";

export function CommonGallery() {
  return (
    <>
      <GallerySection
        name="ExplanationDialog"
        source="src/components/common/ExplanationDialog.tsx"
        purpose="Button that opens a full-width dialog for long explanations or charts."
      >
        <div className="lab-options">
          <ExplanationDialog
            triggerLabel="Open explanation"
            title="Why resample before cross-validation"
            closeLabel="Close explanation"
          >
            <p>
              Dialog body. Pass any long text or chart as children; the title
              and close button are supplied.
            </p>
          </ExplanationDialog>
        </div>
      </GallerySection>

      <GallerySection
        name="ThemeToggle"
        source="src/components/common/ThemeToggle.tsx"
        purpose="Switches between light and dark mode and remembers the choice."
      >
        <div className="lab-options">
          <ThemeToggle />
        </div>
      </GallerySection>

      <GallerySection
        name="Button"
        source="src/components/ui/button.tsx"
        purpose="Base button. Variants: default, outline, ghost. Sizes: default, small, icon."
      >
        <div className="lab-options">
          <Button>Default</Button>
          <Button variant="outline">Outline</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="outline" size="small">
            Small
          </Button>
          <Button variant="outline" disabled>
            Disabled
          </Button>
        </div>
      </GallerySection>
    </>
  );
}

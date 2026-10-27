import type { ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { Maximize2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Button that opens a full-width dialog for long explanations or charts. */
export function ExplanationDialog({
  triggerLabel,
  triggerClassName,
  title,
  closeLabel,
  bodyClassName,
  children,
}: {
  triggerLabel: string;
  triggerClassName?: string;
  title: ReactNode;
  closeLabel: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <Button variant="outline" className={triggerClassName}>
          <Maximize2 size={16} /> {triggerLabel}
        </Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="sheet-overlay" />
        <Dialog.Content
          className="explanation-dialog"
          aria-describedby={undefined}
        >
          <header className="explanation-dialog-heading">
            <Dialog.Title>{title}</Dialog.Title>
            <Dialog.Close asChild>
              <Button variant="ghost" size="icon" aria-label={closeLabel}>
                <X size={20} />
              </Button>
            </Dialog.Close>
          </header>
          <div className={cn("explanation-dialog-body", bodyClassName)}>
            {children}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

"use client";

import { useCallback, useId, useRef, useState } from "react";

import { ACCEPTED_EXTENSIONS } from "@/hooks/use-upload";
import { cn } from "@/lib/utils";

interface VideoDropzoneProps {
  onSelect: (file: File) => void;
  disabled?: boolean;
}

/**
 * Drag-and-drop target with a keyboard-accessible file picker fallback.
 *
 * The visible element is a `<label>` bound to a visually hidden `<input
 * type="file">`. That is what makes the control reachable by keyboard and
 * announced correctly by screen readers — a `<div onClick>` that opens a file
 * dialog looks identical and is unusable without a mouse.
 */
export function VideoDropzone({ onSelect, disabled }: VideoDropzoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Drag events fire on every child element, so a plain boolean flickers as the
  // pointer moves across the zone's contents. Counting enter/leave pairs keeps
  // the highlight steady.
  const dragDepth = useRef(0);

  const handleDragEnter = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      if (disabled) return;
      dragDepth.current += 1;
      setIsDragging(true);
    },
    [disabled],
  );

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) {
      dragDepth.current = 0;
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback(
    (event: React.DragEvent) => {
      // Without this the browser navigates to the dropped file instead of
      // letting the drop handler run.
      event.preventDefault();
      if (!disabled) event.dataTransfer.dropEffect = "copy";
    },
    [disabled],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      dragDepth.current = 0;
      setIsDragging(false);
      if (disabled) return;

      const dropped = event.dataTransfer.files?.[0];
      if (dropped) onSelect(dropped);
    },
    [disabled, onSelect],
  );

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const chosen = event.target.files?.[0];
      if (chosen) onSelect(chosen);
      // Reset so picking the same file twice in a row still fires `change`.
      event.target.value = "";
    },
    [onSelect],
  );

  return (
    <div
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={`${ACCEPTED_EXTENSIONS.join(",")},video/mp4,video/quicktime`}
        onChange={handleChange}
        disabled={disabled}
        className="sr-only"
      />
      <label
        htmlFor={inputId}
        className={cn(
          "border-border/70 bg-card/40 group flex cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors",
          "hover:border-primary/50 hover:bg-card/70",
          "focus-within:ring-ring focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-offset-transparent",
          isDragging && "border-primary bg-primary/5",
          disabled && "pointer-events-none opacity-50",
        )}
      >
        <span
          aria-hidden
          className={cn(
            "bg-muted/60 text-muted-foreground flex size-14 items-center justify-center rounded-full transition-colors",
            isDragging && "bg-primary/15 text-primary",
          )}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="size-6"
          >
            <path d="M12 16V4m0 0L8 8m4-4 4 4" />
            <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
        </span>

        <span className="space-y-1.5">
          <span className="block text-sm font-medium">
            {isDragging ? "Drop your video here" : "Drag a squat video here"}
          </span>
          <span className="text-muted-foreground block text-xs">
            or{" "}
            <span className="text-foreground underline underline-offset-4">
              browse your files
            </span>
          </span>
        </span>

        <span className="text-muted-foreground/80 text-[11px]">
          MP4 or MOV · up to 200 MB · up to 60 seconds
        </span>
      </label>
    </div>
  );
}

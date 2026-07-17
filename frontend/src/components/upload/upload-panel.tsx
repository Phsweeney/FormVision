"use client";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { VideoDropzone } from "@/components/upload/video-dropzone";
import { useUpload } from "@/hooks/use-upload";
import { formatFileSize } from "@/lib/format";

/** Labels for each stage, so the user always knows what is happening. */
const PHASE_LABEL: Record<string, string> = {
  uploading: "Uploading video…",
  starting: "Starting analysis…",
  redirecting: "Opening your results…",
};

/**
 * The upload card: dropzone, preview, progress, and submission.
 *
 * Owns no logic of its own — `useUpload` holds the state machine, leaving this
 * component to render it.
 */
export function UploadPanel() {
  const {
    phase,
    progress,
    error,
    file,
    previewUrl,
    selectFile,
    clear,
    submit,
    isBusy,
  } = useUpload();

  return (
    <div className="space-y-5">
      {!file ? (
        <VideoDropzone onSelect={selectFile} disabled={isBusy} />
      ) : (
        <div className="border-border/70 bg-card/40 space-y-4 rounded-xl border p-4">
          {previewUrl && (
            <video
              key={previewUrl}
              src={previewUrl}
              controls
              playsInline
              // `muted` is required for autoplay-adjacent behaviour and stops a
              // preview surprising the user with sound.
              muted
              className="bg-muted/30 max-h-[380px] w-full rounded-lg object-contain"
            />
          )}

          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{file.name}</p>
              <p className="text-muted-foreground text-xs">
                {formatFileSize(file.size)}
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={clear}
                disabled={isBusy}
              >
                Change
              </Button>
              <Button size="sm" onClick={submit} disabled={isBusy}>
                {isBusy ? "Working…" : "Analyse squat"}
              </Button>
            </div>
          </div>

          {isBusy && (
            <div className="space-y-2 pt-1">
              <div className="text-muted-foreground flex items-center justify-between text-xs">
                <span>{PHASE_LABEL[phase] ?? "Working…"}</span>
                {phase === "uploading" && <span>{progress}%</span>}
              </div>
              <Progress
                // Once the bytes are sent the remaining work has no measurable
                // progress, so the bar sits full while the label explains what
                // is still happening rather than pretending to advance.
                value={phase === "uploading" ? progress : 100}
                className="h-1.5"
              />
            </div>
          )}
        </div>
      )}

      {error && (
        <Alert variant="destructive" role="alert">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}

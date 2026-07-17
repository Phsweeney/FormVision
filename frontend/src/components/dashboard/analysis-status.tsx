import Link from "next/link";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * The processing state.
 *
 * Shows the actual pipeline stages rather than a bare spinner. Pose estimation
 * takes tens of seconds, and naming what is happening is the difference between
 * a wait that feels considered and one that feels broken.
 */
export function ProcessingState({ filename }: { filename?: string }) {
  return (
    <div className="mx-auto w-full max-w-2xl py-16 text-center">
      <div className="border-primary/30 border-t-primary mx-auto size-8 animate-spin rounded-full border-2" />

      <h1 className="mt-6 text-lg font-medium">Analysing your squat</h1>
      <p className="text-muted-foreground mt-1.5 text-sm">
        {filename ? `Processing ${filename}. ` : ""}
        This usually takes under a minute.
      </p>

      <ol className="text-muted-foreground mt-8 space-y-2 text-left text-xs">
        {[
          "Extracting body landmarks from every frame",
          "Measuring knee, hip, and torso angles",
          "Detecting repetitions from hip movement",
          "Rendering the skeleton overlay video",
        ].map((step) => (
          <li key={step} className="flex items-center gap-2.5">
            <span className="bg-muted-foreground/40 size-1 rounded-full" aria-hidden />
            {step}
          </li>
        ))}
      </ol>

      <div className="mt-8 space-y-3" aria-hidden>
        <Skeleton className="h-40 w-full rounded-xl" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-20 rounded-xl" />
          ))}
        </div>
      </div>
    </div>
  );
}

/** The initial load, before the first response arrives. */
export function LoadingState() {
  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 px-6 py-10">
      <Skeleton className="h-8 w-56" />
      <Skeleton className="h-[420px] w-full rounded-xl" />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Array.from({ length: 8 }).map((_, index) => (
          <Skeleton key={index} className="h-24 rounded-xl" />
        ))}
      </div>
    </div>
  );
}

/**
 * A failed or unloadable analysis.
 *
 * The backend guarantees a failed record carries a human-readable message, so
 * it is shown directly rather than replaced with something generic.
 */
export function ErrorState({
  title,
  message,
  onRetry,
}: {
  title: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-16">
      <Alert variant="destructive">
        <AlertTitle>{title}</AlertTitle>
        <AlertDescription>{message}</AlertDescription>
      </Alert>

      <div className="mt-6 flex justify-center gap-3">
        {onRetry && (
          <Button variant="outline" onClick={onRetry}>
            Try again
          </Button>
        )}
        {/* Styled anchor rather than a slotted Button: this shadcn build is on
            Base UI, which has no `asChild`. */}
        <Link href="/" className={buttonVariants()}>
          Upload another video
        </Link>
      </div>
    </div>
  );
}

/**
 * Banner shown when tracking was poor.
 *
 * Deliberately placed above the results rather than beside them: if the
 * measurements are unreliable, the lifter should learn that before reading
 * numbers derived from them, not after.
 */
export function TrackingWarning({ quality }: { quality: number }) {
  return (
    <Alert className="border-amber-500/30 bg-amber-500/5">
      <AlertTitle className="text-amber-400">
        Limited tracking quality
      </AlertTitle>
      <AlertDescription className="text-muted-foreground">
        Your body was clearly visible in only {(quality * 100).toFixed(0)}% of
        frames, so the measurements below are approximate. Re-filming with your
        whole body in frame, in even lighting, will make them more reliable.
      </AlertDescription>
    </Alert>
  );
}

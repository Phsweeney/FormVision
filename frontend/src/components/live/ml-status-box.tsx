import type { MlVerdict } from "@/lib/ml/classify";

/**
 * The live model readout.
 *
 * Deliberately always on screen while the camera is running, unlike the cue
 * banner above it which appears only when the rule-based coach has something to
 * say. This is a status display, not coaching: it answers "what does the model
 * currently think" rather than "what should I change". Nothing here reaches the
 * voice coach.
 *
 * Violet throughout, matching the `ModelBadge` on the upload dashboard, so
 * model output looks the same in both halves of the product and is never
 * mistaken for a measured fact.
 *
 * **The scope of the verdict is the hard part of this component**, not the
 * layout. The model judges three faults and depth is not one of them, so an
 * unqualified "looks correct" on a badly shallow rep is a real claim and a
 * wrong one. Every element here exists to bound what the headline is saying:
 * the tag says it is an experimental classifier rather than a measurement, the
 * scope rows say what was checked and what the camera could not see, and the
 * note says what this box does not judge at all.
 */
export function MlStatusBox({
  verdict,
  available,
  running,
  shallowDepthPercent,
  repInProgress,
}: {
  verdict: MlVerdict | null;
  /**
   * True once the detector bundle is loaded, false when it could not be, and
   * null while the fetch is still in flight. Three states rather than two
   * because the first paint happens before the fetch resolves, and reporting
   * "unavailable" there announces a failure that has not happened.
   */
  available: boolean | null;
  running: boolean;
  /**
   * Depth of the last rep, when the rules judged it shallow. Null otherwise.
   * Measured, not modelled, which is exactly the point of showing it here.
   */
  shallowDepthPercent: number | null;
  /** True while a repetition is being performed, i.e. a verdict is coming. */
  repInProgress: boolean;
}) {
  const showScope = verdict !== null;

  return (
    <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        {/* The tag carries the framing on its own: that this is a model, that
            it classifies, and that it is under test. Enough to stop the panel
            being read as a measurement, without a paragraph under every
            verdict saying so. */}
        <span className="rounded-full border border-violet-400/50 bg-violet-400/10 px-2 py-0.5 text-[10px] font-semibold tracking-wide text-violet-300 uppercase">
          Experimental ML model classification
        </span>
        {verdict ? (
          <>
            {/* Naming the rep is what makes a per-rep verdict legible: without
                it a value that only changes at lockout looks like a stale one. */}
            <span className="text-[10px] font-semibold tracking-widest text-violet-400 uppercase">
              rep {verdict.repIndex}
            </span>
            <span className="ml-auto font-mono text-sm tabular-nums text-violet-300">
              {Math.round(verdict.confidence * 100)}%
            </span>
          </>
        ) : null}
      </div>

      <p className="mt-1 text-lg leading-tight font-semibold">
        {headline(verdict, available, running)}
      </p>

      {showScope ? (
        <dl className="text-muted-foreground mt-2 space-y-0.5 text-xs">
          <Row label="checked" values={verdict.checking} empty="nothing yet" />
          {/* Naming what the camera cannot see is the point. "No fault
              detected" otherwise means either "I checked and it is fine" or
              "I could not see the thing that is wrong". */}
          {verdict.notChecking.length > 0 ? (
            <Row label="blind to" values={verdict.notChecking} />
          ) : null}
        </dl>
      ) : (
        <p className="text-muted-foreground mt-2 text-xs leading-relaxed">
          {detail(available, running)}
        </p>
      )}

      {/* The verdict deliberately holds still through the rep, so without this
          the box looks frozen while the lifter is mid-squat. */}
      {running && repInProgress ? (
        <p className="mt-2 text-xs text-violet-400/80">
          Watching this rep, verdict at lockout...
        </p>
      ) : null}

      {showScope ? (
        <div className="mt-2.5 border-t border-violet-500/20 pt-2">
          {shallowDepthPercent !== null ? (
            <p className="text-xs leading-relaxed text-amber-300">
              Last rep reached {Math.round(shallowDepthPercent)}% depth, which is
              short. <span className="font-medium">The verdict above does not
              cover depth.</span> See the depth readings below.
            </p>
          ) : (
            <p className="text-muted-foreground text-xs leading-relaxed">
              Depth and torso lean are measured directly, not judged by this
              model. A clean verdict here says nothing about how deep you went.
            </p>
          )}
        </div>
      ) : null}
    </div>
  );
}

function Row({
  label,
  values,
  empty = "nothing visible",
}: {
  label: string;
  values: string[];
  empty?: string;
}) {
  return (
    <div className="flex gap-1.5">
      <dt className="shrink-0">{label}:</dt>
      <dd className="text-foreground/70">
        {values.length > 0 ? values.join(", ") : empty}
      </dd>
    </div>
  );
}

function headline(
  verdict: MlVerdict | null,
  available: boolean | null,
  running: boolean,
): string {
  if (available === null) return "Loading model";
  if (!available) return "Model unavailable";
  if (!running) return "Not running";
  if (!verdict) return "Warming up";
  if (verdict.fault) return verdict.fault.label;
  // A rep the camera could see nothing of is not a clean rep, and saying so is
  // the difference between "I checked" and "I could not look".
  if (verdict.checking.length === 0) return "Nothing visible from this angle";
  // "No fault detected" rather than "Looks correct": the model checks three
  // specific faults, and a squat can be poor in ways none of them cover. The
  // narrower phrase is the true one.
  return "No fault detected";
}

function detail(available: boolean | null, running: boolean): string {
  if (available === null) return "Fetching the trained squat fault detectors.";
  if (!available) {
    return (
      "No detector bundle was loaded, so the rest of live coaching is running " +
      "without it. Run `python -m training.export_web` and `npm run setup:live`."
    );
  }
  if (!running) return "Start the camera to see what the model makes of your squat.";
  // The model ranks each rep against the rest of the session, so the first one
  // establishes the range of motion and is not itself judged. Verdicts start
  // from the second rep.
  return "Your first rep sets the range of motion to compare against. Verdicts start from rep 2.";
}

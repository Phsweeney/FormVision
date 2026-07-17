import { UploadPanel } from "@/components/upload/upload-panel";

/** What the analysis produces, shown so the value is clear before uploading. */
const CAPABILITIES = [
  {
    title: "Pose tracking",
    body: "Body landmarks extracted from every frame and rendered as a skeleton overlay on your video.",
  },
  {
    title: "Rep counting",
    body: "Repetitions detected from hip movement, with each one broken into its descent and ascent.",
  },
  {
    title: "Joint angles",
    body: "Knee, hip, and torso angles measured frame by frame and plotted so you can see where form changed.",
  },
  {
    title: "Coaching feedback",
    body: "Depth, forward lean, left-right balance, consistency, and tempo — each with an explanation of why it matters.",
  },
] as const;

/**
 * Landing page.
 *
 * A Server Component. Only the upload panel is interactive, so it is the only
 * part of this page shipped to the browser as JavaScript.
 */
export default function HomePage() {
  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-14 sm:py-20">
      <section className="space-y-4 text-center">
        <span className="border-border/70 bg-card/60 text-muted-foreground inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs">
          <span className="bg-primary/80 size-1.5 rounded-full" aria-hidden />
          Back squat · V1
        </span>

        <h1 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
          See what your squat actually looks like
        </h1>

        <p className="text-muted-foreground mx-auto max-w-xl text-sm text-pretty sm:text-base">
          Upload a video and FormVision measures your depth, counts your reps,
          tracks your joint angles, and tells you what to work on — using
          explicit biomechanical rules, not a black box.
        </p>
      </section>

      <section className="mt-10">
        <UploadPanel />
      </section>

      <section className="mt-8">
        <div className="border-border/60 bg-card/30 rounded-xl border p-5">
          <h2 className="text-sm font-medium">Filming tips</h2>
          <ul className="text-muted-foreground mt-3 grid gap-2 text-xs sm:grid-cols-2">
            <li>· Keep your whole body in frame for the entire set.</li>
            <li>· Film side-on for the most accurate depth and lean.</li>
            <li>· Film front-on to check left-right balance.</li>
            <li>· One person in shot, in even lighting.</li>
          </ul>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="text-muted-foreground mb-4 text-xs font-medium tracking-wide uppercase">
          What you get
        </h2>
        <dl className="border-border/60 grid gap-px overflow-hidden rounded-xl border sm:grid-cols-2">
          {CAPABILITIES.map((item) => (
            <div key={item.title} className="bg-card/40 p-5">
              <dt className="text-sm font-medium">{item.title}</dt>
              <dd className="text-muted-foreground mt-1.5 text-xs leading-relaxed">
                {item.body}
              </dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
}

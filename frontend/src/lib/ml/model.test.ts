import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { interpolateIsotonic, parseBundle, scoreFault } from "./model";
import type { FaultModelBundle } from "./model";

/**
 * The artifacts live in the backend because that is where they are produced.
 * Read from there rather than from `public/`, so the test exercises the real
 * exported files and does not quietly pass against a stale staged copy.
 */
const ARTIFACTS = join(
  process.cwd(),
  "..",
  "backend",
  "app",
  "ml",
  "artifacts",
);

function readJson(name: string): unknown {
  return JSON.parse(readFileSync(join(ARTIFACTS, name), "utf8"));
}

const bundle = parseBundle(readJson("squat_faults_web.json")) as FaultModelBundle;

interface FixtureEntry {
  faultId: string;
  features: (number | null)[][];
  expected: number[];
}
const fixture = readJson("squat_faults_web_fixture.json") as FixtureEntry[];

/**
 * The test that justifies having two implementations of one model.
 *
 * A hand-written port of a fitted classifier is a standing invitation to drift:
 * a wrong comparison operator, a missed sign, a scaler applied in the wrong
 * order. None of those would throw, and all of them would quietly produce
 * plausible-looking probabilities. Pinning the TypeScript output to Python's on
 * generated rows is what converts "should be equivalent" into "is equivalent",
 * and it fails loudly the moment either side changes.
 */
describe("parity with scikit-learn", () => {
  it("has a fixture covering every shipped detector", () => {
    expect(fixture.length).toBeGreaterThan(0);
    for (const entry of fixture) {
      expect(bundle.faults[entry.faultId]).toBeDefined();
      expect(entry.features.length).toBeGreaterThan(100);
    }
  });

  for (const entry of fixture) {
    it(`reproduces Python's probabilities for ${entry.faultId}`, () => {
      const model = bundle.faults[entry.faultId];
      let worst = 0;

      for (let row = 0; row < entry.features.length; row++) {
        const actual = scoreFault(model, entry.features[row]);
        worst = Math.max(worst, Math.abs(actual - entry.expected[row]));
      }

      // 1e-6 is the rounding the exporter applies to the weights, so anything
      // larger is a genuine disagreement rather than lost precision.
      expect(worst).toBeLessThan(1e-6);
    });
  }

  it("covers the imputation path", () => {
    // The exporter deliberately blanks a tenth of the fixture, because a frame
    // that lost a leg still has to score. If that stopped happening the parity
    // claim would quietly narrow.
    const missing = fixture
      .flatMap((entry) => entry.features)
      .flat()
      .filter((value) => value === null);
    expect(missing.length).toBeGreaterThan(0);
  });
});

describe("bundle validation", () => {
  it("accepts the real bundle", () => {
    expect(bundle).not.toBeNull();
    expect(Object.keys(bundle.faults).length).toBeGreaterThan(0);
  });

  it("rejects anything malformed rather than throwing", () => {
    // The caller is a camera loop, so a corrupt model must cost the readout and
    // nothing else.
    expect(parseBundle(null)).toBeNull();
    expect(parseBundle("nonsense")).toBeNull();
    expect(parseBundle({})).toBeNull();
    expect(parseBundle({ faults: { x: { features: [] } } })).toBeNull();
  });
});

describe("isotonic interpolation", () => {
  const calibrator = { x: [0, 1, 2], y: [0.1, 0.5, 0.9] };

  it("interpolates linearly between knots", () => {
    expect(interpolateIsotonic(calibrator, 0.5)).toBeCloseTo(0.3, 10);
    expect(interpolateIsotonic(calibrator, 1.5)).toBeCloseTo(0.7, 10);
  });

  it("returns the knot value exactly at a knot", () => {
    expect(interpolateIsotonic(calibrator, 1)).toBeCloseTo(0.5, 10);
  });

  it("clamps beyond both ends", () => {
    // scikit-learn holds the end values rather than extrapolating, which would
    // otherwise produce probabilities outside [0, 1].
    expect(interpolateIsotonic(calibrator, -99)).toBeCloseTo(0.1, 10);
    expect(interpolateIsotonic(calibrator, 99)).toBeCloseTo(0.9, 10);
  });

  it("survives an empty calibrator", () => {
    expect(interpolateIsotonic({ x: [], y: [] }, 0.5)).toBe(0);
  });
});

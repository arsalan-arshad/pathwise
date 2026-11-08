import type { KnobSpace, Plan } from "./types.js";

/** Stable string id for a plan, independent of key order. */
export function planKey(plan: Plan): string {
  return Object.keys(plan)
    .sort()
    .map((k) => `${k}=${String(plan[k])}`)
    .join("|");
}

/** Every combination of knob values (the cartesian product of the space). */
export function enumeratePlans(space: KnobSpace): Plan[] {
  const knobs = Object.keys(space);
  let plans: Plan[] = [{}];
  for (const knob of knobs) {
    const values = space[knob];
    if (!values || values.length === 0) {
      throw new Error(`Knob "${knob}" has no values`);
    }
    const next: Plan[] = [];
    for (const partial of plans) {
      for (const v of values) next.push({ ...partial, [knob]: v });
    }
    plans = next;
  }
  return plans;
}

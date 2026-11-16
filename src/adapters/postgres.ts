/**
 * Postgres store for the planner's learned data: zero dependencies, no `pg` import. You pass in a client that has
 * `query(text, params)` (node-postgres `Pool` or `Client`, or anything pg-compatible), so this file never opens a
 * connection, never sees a connection string and never reads the environment.
 *
 *   import { Pool } from "pg";
 *   const store = postgresStore({ client: new Pool({ connectionString: process.env.DATABASE_URL }) });
 *   await store.init();                       // CREATE TABLE IF NOT EXISTS (once)
 *   await store.save("support-bot", planner); // upsert planner.exportState() under that key
 *   await store.load("support-bot", freshPlanner); // freshPlanner.importState(...); returns the record count, or undefined
 *
 * What is stored is exactly `planner.exportState()` (the calibration and feedback records). `importState()` APPENDS, so load
 * into a freshly created planner (the planner recomputes profiles and thresholds from the records). Table name: letters,
 * digits and underscores only (validated, then double-quoted); the key and the state are always bound parameters, never
 * spliced into SQL. Errors from the client are rewrapped; `postgres://` URLs and `user:password@host` fragments are removed from their text, but a bare host name or user name a driver prints on its own (e.g. getaddrinfo ENOTFOUND host) is not recognised.
 */
import type { QueryRecord } from "../types.js";

/** The only thing used from a pg client. */
export interface PgLikeClient {
  query(text: string, params?: unknown[]): Promise<{ rows: Array<Record<string, unknown>> }>;
}

/** What the store needs from a planner (structural, so a Planner fits). */
export interface StatefulPlanner {
  exportState(): QueryRecord[];
  importState(records: QueryRecord[]): void;
}

export interface PostgresStoreOptions {
  client: PgLikeClient;
  /** Default "pathwise_state". Must match ^[A-Za-z_][A-Za-z0-9_]*$ and be at most 63 characters. */
  table?: string;
}

export interface PostgresStore {
  /** CREATE TABLE IF NOT EXISTS. Call once at startup. */
  init(): Promise<void>;
  /** Upserts the planner's exported state under `key`. */
  save(key: string, planner: Pick<StatefulPlanner, "exportState">): Promise<void>;
  /** Imports the state saved under `key` into `planner`. Returns the number of records, or undefined when the key does not exist. */
  load(key: string, planner: Pick<StatefulPlanner, "importState">): Promise<number | undefined>;
}

const TABLE_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const MAX_KEY = 512;

function scrub(err: unknown): string {
  const text = (err instanceof Error ? err.message : String(err)).slice(0, 2000); // bound the input BEFORE the regexes (the second is quadratic on long text without ':')
  return text
    .replace(/postgres(?:ql)?:\/\/[^\s'"]+/gi, "[redacted-url]")
    .replace(/[^\s:@/'"]+:[^\s@/'"]+@[^\s/'"]+/g, "[redacted-credentials]") // scheme-less user:password@host
    .slice(0, 300);
}

function checkKey(key: string) {
  if (typeof key !== "string" || key.length === 0 || key.length > MAX_KEY) throw new Error(`postgresStore: key must be a non-empty string of at most ${MAX_KEY} characters`);
}

export function postgresStore(options: PostgresStoreOptions): PostgresStore {
  const { client } = options;
  if (!client || typeof client.query !== "function") throw new Error("postgresStore: client with a query(text, params) method is required");
  const table = options.table ?? "pathwise_state";
  if (!TABLE_RE.test(table) || table.length > 63) throw new Error("postgresStore: table must match ^[A-Za-z_][A-Za-z0-9_]*$ and be at most 63 characters");
  const t = `"${table}"`; // validated above, so quoting is for reserved words only
  const run = async (what: string, text: string, params?: unknown[]) => {
    try {
      return await client.query(text, params);
    } catch (err) {
      throw new Error(`postgresStore: ${what} failed: ${scrub(err)}`);
    }
  };
  return {
    async init() {
      await run("init", `CREATE TABLE IF NOT EXISTS ${t} (key text PRIMARY KEY, state jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())`);
    },
    async save(key, planner) {
      checkKey(key);
      const json = JSON.stringify(planner.exportState());
      await run("save", `INSERT INTO ${t} (key, state) VALUES ($1, $2::jsonb) ON CONFLICT (key) DO UPDATE SET state = EXCLUDED.state, updated_at = now()`, [key, json]);
    },
    async load(key, planner) {
      checkKey(key);
      const res = await run("load", `SELECT state FROM ${t} WHERE key = $1`, [key]);
      if (!res.rows || res.rows.length === 0) return undefined;
      const raw = res.rows[0].state;
      let records: unknown;
      try {
        records = typeof raw === "string" ? JSON.parse(raw) : raw; // pg returns jsonb already parsed; a string is parsed here
      } catch {
        throw new Error("postgresStore: the stored state is not valid JSON");
      }
      if (!Array.isArray(records) || records.some((r) => !r || !Array.isArray((r as QueryRecord).features) || !Array.isArray((r as QueryRecord).observations))) {
        throw new Error("postgresStore: the stored state is not a list of planner records (expected features and observations on each)");
      }
      planner.importState(records as QueryRecord[]);
      return records.length;
    },
  };
}

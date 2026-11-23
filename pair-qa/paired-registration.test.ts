import { execFile } from "node:child_process";
import { randomBytes } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { expect, it } from "vitest";
import { DaemonControlApiServer } from "@claudexor/control-api";
import { sqlFixture } from "../../daemon/src/store/test-support/sql-fixture.js";
import { projectRegistryServices } from "./project-registry-services.js";

// This file is staged at claudexor/packages/cli/src/ on the disposable runner.
const workspace = resolve(import.meta.dirname, "../../../..");
const hostCheckout = join(workspace, "ouroboros");
const driver = join(workspace, "qa/pair-qa/positive.py");

function runGateway(dir: string, runnerTemp: string, fixture: object): Promise<void> {
  return new Promise((done, reject) => {
    const child = execFile(
      "python3",
      ["-I", "-B", "-c", "import runpy, sys; sys.path.insert(0, '.'); runpy.run_path(sys.argv[1], run_name='__main__')", driver],
      {
        cwd: hostCheckout,
        env: { PATH: process.env.PATH, HOME: dir, RUNNER_TEMP: runnerTemp, PAIR_QA_JOB: "github-hosted" },
        timeout: 35_000,
      },
      (error, stdout, stderr) => {
        // No subprocess exception, argv, HTTP content or arbitrary stderr in CI output.
        if (!error && stdout.trim() ===
          "paired custody: unsent A; B pending -> fixture terminal; release POST=1; repeat=0") return done();
        const diagnostic = stderr.trim().split("\n").at(-1) ?? "";
        if (/^paired gateway failed at [a-z_]+ \([A-Za-z_][A-Za-z_0-9]*\)$/.test(diagnostic)) {
          reject(new Error(diagnostic));
        } else {
          reject(new Error("paired gateway subprocess failed (no safe diagnostic)"));
        }
      },
    );
    child.stdin?.on("error", () => { /* execFile reports process failure. */ });
    child.stdin?.end(JSON.stringify(fixture) + "\n");
  });
}

it("keeps a registry entry for the pending adopter, then releases it once after settlement", async () => {
  if (process.platform !== "linux" || process.env.GITHUB_ACTIONS !== "true" ||
    process.env.RUNNER_ENVIRONMENT !== "github-hosted" || process.env.PAIR_QA_JOB !== "github-hosted" ||
    !process.env.RUNNER_TEMP) {
    throw new Error("paired fixture requires its disposable GitHub-hosted workflow job");
  }
  const runnerTemp = realpathSync(process.env.RUNNER_TEMP);
  const dir = realpathSync(mkdtempSync(join(runnerTemp, "cx-pair-")));
  const root = join(dir, "project + &");
  const marker = join(root, "sentinel.txt");
  const token = randomBytes(32).toString("hex");
  let sql: Awaited<ReturnType<typeof sqlFixture>> | undefined;
  let server: DaemonControlApiServer | undefined;
  try {
    mkdirSync(root);
    writeFileSync(marker, "fixture\n");
    writeFileSync(join(dir, "fixture-generation"), "fixture generation\n");
    const own = (sql = await sqlFixture(dir));
    const services = projectRegistryServices(() => own.graph.projects);
    const observed: string[] = [];
    const registrations: Array<{ id: string; created: boolean; idempotencyKey: string }> = [];
    const inspections: Array<{ state: string; sql: unknown; releases: number }> = [];
    const releaseResults: string[] = [];
    const projectRows = () => own.store.prepare("SELECT id, status FROM project WHERE root = ? ORDER BY id").all(root);
    const journalMutations = () => own.store.prepare(
      "SELECT type FROM event WHERE type IN ('project.registered', 'project.unregistered') ORDER BY pid, seq",
    ).all();

    server = new DaemonControlApiServer({
      host: "127.0.0.1",
      port: 0,
      token,
      // The server owns only the real project services below; it has no run/agent backend.
      daemon: {} as never,
      storeId: () => own.store.storeId,
      services: {
        ...services,
        registerProject: async (input) => {
          observed.push("POST /projects");
          if (input.root !== root) throw new Error("unexpected registration root");
          const result = await services.registerProject(input);
          registrations.push({ id: result.id, created: result.created, idempotencyKey: input.idempotencyKey });
          return result;
        },
        inspectProjectRegistration: async (id, binding, candidate) => {
          observed.push("GET /projects/:id/registration");
          if (binding.expectedRoot !== root || id !== registrations[0]?.id) {
            throw new Error("unexpected inspection binding");
          }
          const result = await services.inspectProjectRegistration(id, binding, candidate);
          inspections.push({ state: result.state, sql: projectRows(), releases: releaseResults.length });
          return result;
        },
        releaseProject: async (id, binding) => {
          observed.push("POST /projects/:id/release");
          if (binding.expectedRoot !== root || id !== registrations[0]?.id || inspections.length !== 1) {
            throw new Error("unexpected release binding or phase");
          }
          const result = await services.releaseProject(id, binding);
          releaseResults.push(result.state);
          return result;
        },
        removeProject: async () => {
          observed.push("DELETE /projects/:id");
          throw new Error("legacy deletion is not part of the paired fixture");
        },
      },
    });
    const { port } = await server.start();
    await runGateway(dir, runnerTemp, { root, token, port });
    expect(observed).toEqual([
      "POST /projects", "POST /projects", "GET /projects/:id/registration",
      "POST /projects/:id/release", "GET /projects/:id/registration",
    ]);
    expect(registrations).toHaveLength(2);
    const [created, adopted] = registrations;
    expect(Boolean(created?.id)).toBe(true);
    const id = created!.id;
    expect([created?.created, adopted?.created, adopted?.id]).toEqual([true, false, id]);
    expect(new Set(registrations.map((registration) => registration.idempotencyKey)).size).toBe(2);
    const archived = [{ id, status: "archived" }];
    expect(inspections).toEqual([
      { state: "present", sql: [{ id, status: "active" }], releases: 0 },
      { state: "absent", sql: archived, releases: 1 },
    ]);
    expect(releaseResults).toEqual(["released"]);
    expect(projectRows()).toEqual(archived);
    expect(journalMutations()).toEqual([
      { type: "project.registered" }, // A creates the row
      { type: "project.registered" }, // B's distinct idempotency key adopts it
      { type: "project.unregistered" },
    ]);
    expect(own.graph.projects.get(id) == null).toBe(true);
    expect(own.records()).toEqual([]);
    expect(readFileSync(marker, "utf8")).toBe("fixture\n");
    console.log("paired registry: 2 registration POSTs, active while B pending, 1 bound release, archived, sentinel intact");
  } finally {
    // Keep the fixture directory if the server or store could not close.
    await server?.stop();
    await sql?.close();
    rmSync(dir, { recursive: true, force: true });
  }
}, 60_000);

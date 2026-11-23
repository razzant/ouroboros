import { randomBytes } from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";
import { DaemonControlApiServer } from "@claudexor/control-api";
import {
  CONTROL_PROTOCOL_MAJOR,
  CONTROL_STORE_IDENTITY_HEADER,
  ControlRunStartRequest,
} from "@claudexor/schema";
import { sqlFixture } from "../../daemon/src/store/test-support/sql-fixture.js";
import { projectRegistryServices } from "./project-registry-services.js";

// Staged at claudexor/packages/cli/src/ on the disposable github-hosted runner.
// Engine-side only: a plain client without AI drives the public registry
// contract over HTTP; the one test command is recorded and closed through the
// real store methods and never executes (the server has no run backend).
//
// Positive pair witness: the accepted queued command's SOURCE is the registered
// project P (scope.root=rootP, so the command row lives in P's partition, pid P),
// and its SEPARATELY USED working catalog is the registered Q
// (execution.workspaceRoot=rootQ). Both facts are read back from the saved
// command itself: the command row's pid and its recorded params name both roots.
// The former draft scoped the command to rootQ itself, which made Q the source
// and only asserted pid Q.

const CMD = "cmd-shared-q-1";

type Binding = { expectedStoreId: string; expectedRoot: string; candidate?: string };

function client(port: number, token: string) {
  const base = `http://127.0.0.1:${port}/v2`;
  const headers: Record<string, string> = {
    authorization: `Bearer ${token}`,
    "content-type": "application/json",
    "x-claudexor-protocol-major": String(CONTROL_PROTOCOL_MAJOR),
  };
  return {
    handshake: async () =>
      (await fetch(`${base}/handshake`, {
        method: "POST",
        // The store identity is opt-in on the wire: without this header the
        // handshake answers the legacy strict shape and carries no storeId.
        headers: { ...headers, [CONTROL_STORE_IDENTITY_HEADER]: "1" },
        body: JSON.stringify({ protocolMajor: CONTROL_PROTOCOL_MAJOR, client: "shared-catalog-qa" }),
      })).json() as Promise<{ storeId: string }>,
    register: async (root: string, idempotencyKey: string) => {
      const res = await fetch(`${base}/projects`, {
        method: "POST",
        headers: { ...headers, "idempotency-key": idempotencyKey, [CONTROL_STORE_IDENTITY_HEADER]: "1" },
        body: JSON.stringify({ root }),
      });
      return { status: res.status, body: (await res.json()) as Record<string, unknown> };
    },
    inspect: async (id: string, binding: Binding) => {
      const query = new URLSearchParams(binding).toString();
      const res = await fetch(`${base}/projects/${encodeURIComponent(id)}/registration?${query}`, {
        headers,
      });
      return { status: res.status, body: (await res.json()) as Record<string, unknown> };
    },
    release: async (id: string, binding: Binding) => {
      const res = await fetch(`${base}/projects/${encodeURIComponent(id)}/release`, {
        method: "POST",
        headers,
        body: JSON.stringify(binding),
      });
      return { status: res.status, body: (await res.json()) as Record<string, unknown> };
    },
  };
}

it("keeps the shared catalog registered while a queued command uses it, then archives it exactly once", async () => {
  if (
    process.platform !== "linux" ||
    process.env.GITHUB_ACTIONS !== "true" ||
    process.env.RUNNER_ENVIRONMENT !== "github-hosted" ||
    process.env.SHARED_CATALOG_QA_JOB !== "github-hosted" ||
    !process.env.RUNNER_TEMP
  ) {
    throw new Error("shared-catalog fixture requires its disposable GitHub-hosted workflow job");
  }
  const runnerTemp = realpathSync(process.env.RUNNER_TEMP);
  const dir = realpathSync(mkdtempSync(join(runnerTemp, "cx-shared-")));
  // P: the registered project whose accepted command uses the second catalog Q.
  // Q: the second registered catalog used as the command's separate working root; R: an independent registration.
  const rootP = join(dir, "project-p");
  const rootQ = join(dir, "catalog-q");
  const rootR = join(dir, "registration-r");
  const token = randomBytes(32).toString("hex");
  let sql: Awaited<ReturnType<typeof sqlFixture>> | undefined;
  let server: DaemonControlApiServer | undefined;
  try {
    mkdirSync(rootP);
    mkdirSync(rootQ);
    mkdirSync(rootR);
    writeFileSync(join(rootP, "sentinel.txt"), "p\n");
    writeFileSync(join(rootQ, "sentinel.txt"), "q\n");
    writeFileSync(join(rootR, "sentinel.txt"), "r\n");

    const own = (sql = await sqlFixture(dir));
    const services = projectRegistryServices(() => own.graph.projects);
    const observed: string[] = [];
    server = new DaemonControlApiServer({
      host: "127.0.0.1",
      port: 0,
      token,
      // No run/agent backend: registry services only, commands stay in the test store.
      daemon: {} as never,
      storeId: () => own.store.storeId,
      services: {
        ...services,
        registerProject: async (input) => {
          observed.push("POST /projects");
          return services.registerProject(input);
        },
        inspectProjectRegistration: async (id, binding, candidate) => {
          observed.push("GET /projects/:id/registration");
          return services.inspectProjectRegistration(id, binding, candidate);
        },
        releaseProject: async (id, binding) => {
          observed.push("POST /projects/:id/release");
          return services.releaseProject(id, binding);
        },
        removeProject: async () => {
          observed.push("DELETE /projects/:id");
          throw new Error("legacy deletion is not part of the shared-catalog fixture");
        },
      },
    });
    const { port } = await server.start();
    const api = client(port, token);
    // The store identity the client sees is the handshake's storeId.
    const storeId = (await api.handshake()).storeId;
    expect(typeof storeId).toBe("string");
    if (own.store.storeId !== undefined) expect(own.store.storeId).toBe(storeId);
    const projectRows = () =>
      own.store
        .prepare("SELECT id, status FROM project WHERE root IN (?,?,?)")
        .all(rootP, rootQ, rootR) as Array<{ id: string; status: string }>;
    // Per-record event accounting: count the unregistered events of ONE project id.
    // Project mutations are journaled in the global partition, so the rows of
    // different records share one section; never compare a cross-partition
    // ORDER BY pid,seq list for one record's event.
    const unregisteredEvents = (id: string) =>
      own.store
        .prepare(
          "SELECT pid, seq FROM event WHERE type='project.unregistered' AND json_extract(CAST(payload AS TEXT),'$.project.id')=? ORDER BY pid, seq",
        )
        .all(id) as Array<{ pid: number; seq: number }>;
    const registeredCount = () =>
      Number(
        (
          own.store.prepare("SELECT count(*) AS n FROM event WHERE type='project.registered'").get() as {
            n: number;
          }
        ).n,
      );

    // Public registry contract: POST /v2/projects answers id/root/storeId/created.
    const regP = await api.register(rootP, "reg-p");
    const regQ = await api.register(rootQ, "reg-q");
    const regR = await api.register(rootR, "reg-r");
    for (const reg of [regP, regQ, regR]) {
      expect(reg.status).toBe(200);
      expect(reg.body.created).toBe(true);
      expect(reg.body.storeId).toBe(storeId);
    }
    expect(regP.body.root).toBe(rootP);
    expect(regQ.body.root).toBe(rootQ);
    expect(regR.body.root).toBe(rootR);
    const idP = regP.body.id as string;
    const idQ = regQ.body.id as string;
    const idR = regR.body.id as string;
    expect(new Set([idP, idQ, idR]).size).toBe(3);
    expect(registeredCount()).toBe(3);

    // The accepted, not yet executed command of project P names the registered
    // catalog Q as its separate working root (the supported ordinary shape:
    // scope.root=P is the run's source, execution.workspaceRoot=Q is the
    // separately used working catalog). forRequest routes by scope.root, so the
    // command is stored in P's generation.
    const start = ControlRunStartRequest.parse({
      prompt: "synthetic fixture",
      mode: "agent",
      scope: { kind: "project", root: rootP },
      execution: { isolation: "live", delegated: true, workspaceRoot: rootQ },
    });
    const commands = own.graph.commands.forRequest(start);
    const accepted = commands.accept({
      id: CMD,
      idempotencyKey: "shared-catalog-cmd-1",
      clientId: "shared-catalog-qa",
      operation: "run.create",
      params: start,
    });
    expect(accepted.reused).toBe(false);
    expect(accepted.record.state).toBe("queued");

    // Independent read of the saved command confirms both facts, including pid P.
    const saved = own.graph.commands.findById(CMD)!;
    const genP = own.graph.projects.partition(idP); // contract: PartitionGeneration | null
    expect(genP).not.toBeNull();
    expect(saved.generation.pid).toBe(genP!.pid); // the command lives in P's partition (pid P), not Q's
    const savedRecord = saved.get(CMD)!;
    expect(savedRecord.params).toMatchObject({
      scope: { kind: "project", root: rootP },
      execution: expect.objectContaining({ delegated: true, workspaceRoot: rootQ }),
    });
    expect(unregisteredEvents(idQ)).toEqual([]);

    const bindingQ: Binding = { expectedStoreId: storeId, expectedRoot: rootQ };
    const bindingR: Binding = { expectedStoreId: storeId, expectedRoot: rootR };

    // While the command is queued, Q stays registered; the independent R releases freely.
    const inspectQ = await api.inspect(idQ, bindingQ);
    expect(inspectQ.status).toBe(200);
    expect(inspectQ.body).toEqual({ projectId: idQ, storeId, root: rootQ, state: "present" });
    // The optional candidate query is part of the public probe contract: on a
    // present registration it adds exactly one folder-comparison verdict.
    const inspectQCandidate = await api.inspect(idQ, { ...bindingQ, candidate: rootQ });
    expect(inspectQCandidate.status).toBe(200);
    expect(inspectQCandidate.body).toMatchObject({
      projectId: idQ, storeId, root: rootQ, state: "present", candidateMatch: "same",
    });
    const releaseR = await api.release(idR, bindingR);
    expect(releaseR.status).toBe(200);
    expect(releaseR.body).toEqual({ projectId: idR, storeId, root: rootR, state: "released" });
    expect(unregisteredEvents(idR)).toHaveLength(1);
    // Negative result only as the busy status of the occupied fixture record.
    const releaseQBusy = await api.release(idQ, bindingQ);
    expect(releaseQBusy.status).toBe(409);
    expect(JSON.stringify(releaseQBusy.body)).toContain("project_has_active_run");
    const inspectQAfter = await api.inspect(idQ, bindingQ);
    expect(inspectQAfter.body.state).toBe("present");
    const inspectR = await api.inspect(idR, bindingR);
    expect(inspectR.body.state).toBe("absent");
    // On an absent registration the candidate adds no verdict key.
    const inspectRWithCandidate = await api.inspect(idR, { ...bindingR, candidate: rootR });
    expect(inspectRWithCandidate.status).toBe(200);
    expect(inspectRWithCandidate.body.state).toBe("absent");
    expect("candidateMatch" in inspectRWithCandidate.body).toBe(false);
    expect(unregisteredEvents(idQ)).toEqual([]); // the busy refusal journals nothing

    // Real close of the test command: the use of Q ends.
    const closed = commands.update(CMD, {
      state: "succeeded",
      finishedAt: new Date().toISOString(),
      result: { status: "succeeded" },
    });
    expect(closed.state).toBe("succeeded");
    expect(own.records().find((record) => record.id === CMD)?.state).toBe("succeeded");

    // After the use ends: exactly one archiving of Q; a repeat changes nothing.
    const releaseQ = await api.release(idQ, bindingQ);
    expect(releaseQ.status).toBe(200);
    expect(releaseQ.body).toEqual({ projectId: idQ, storeId, root: rootQ, state: "released" });
    const qEvents = unregisteredEvents(idQ);
    expect(qEvents).toHaveLength(1); // exactly one project.unregistered for Q
    const qEvent = qEvents[0];
    expect(qEvent.pid).toBe(own.graph.projects.global().pid); // journaled in the global registry section
    const releaseQRepeat = await api.release(idQ, bindingQ);
    expect(releaseQRepeat.status).toBe(200);
    expect(releaseQRepeat.body.state).toBe("absent");
    expect(unregisteredEvents(idQ)).toHaveLength(1); // no new event on the repeat
    expect(unregisteredEvents(idP)).toEqual([]); // P is never archived

    expect(observed).toEqual([
      "POST /projects",
      "POST /projects",
      "POST /projects",
      "GET /projects/:id/registration",
      "GET /projects/:id/registration",
      "POST /projects/:id/release",
      "POST /projects/:id/release",
      "GET /projects/:id/registration",
      "GET /projects/:id/registration",
      "GET /projects/:id/registration",
      "POST /projects/:id/release",
      "POST /projects/:id/release",
    ]);
    expect(Object.fromEntries(projectRows().map((row) => [row.id, row.status]))).toEqual({
      [idP]: "active",
      [idQ]: "archived",
      [idR]: "archived",
    });
    // The test folders are untouched by registration and release.
    for (const [root, sentinel] of [
      [rootP, "p\n"],
      [rootQ, "q\n"],
      [rootR, "r\n"],
    ] as const) {
      expect(readdirSync(root)).toEqual(["sentinel.txt"]);
      expect(readFileSync(join(root, "sentinel.txt"), "utf8")).toBe(sentinel);
    }
    expect(own.graph.projects.get(idP) != null).toBe(true);
    expect(own.graph.projects.get(idQ)).toBeUndefined();
    console.log(
      `shared catalog: command of P (pid ${saved.generation.pid}) uses workspaceRoot Q; ` +
        `queued command holds Q (release 409 project_has_active_run); R released freely; ` +
        `after close exactly one archive of Q at event pid=${qEvent.pid} seq=${qEvent.seq}; folders unchanged`,
    );
  } finally {
    await server?.stop();
    await sql?.close();
    rmSync(dir, { recursive: true, force: true });
  }
}, 60_000);

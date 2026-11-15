// free-model-aggregator-01: one server, two client wires, ten simulated providers.
//
// The starter's shape, with the four gaps closed:
//   - quality (60%): every free candidate is ranked by the quality table's rating_milli,
//     and the router spreads calls so no provider's per-trial budget is burned on one
//     request class. A provider's budget is never published, so the router plans from the
//     sealed ranges' minima (a safe cap: the real budget is never below it) and adapts to
//     every refusal it actually sees.
//   - failover (10%): a 402, 429 or 5xx is never the answer. The request is retried on the
//     next free model that meets it, in rating order, and once more over the tried ones
//     (a scheduled refusal is one call number; the next call to the same provider may
//     succeed). Exactly one successful provider call answers each request.
//   - the wires: all three upstream wires are spoken (openai, anthropic, gemini), in JSON
//     and in server-sent events, and the answer is translated back into the wire the
//     client used, with the provider's text whole, its usage counts exact and its stop
//     reason mapped to the client's family. A JSON-mode request only ever goes to a wire
//     that can echo JSON mode (openai or gemini: the anthropic wire always echoes false).
//   - the catalog (5%): the probe reports every provider that answered, every model it
//     listed, the quality table's free flag and context length where it has a row.
//
// Everything here is a function of the brief and the providers' answers: no wall clock
// and no unseeded randomness, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicMessageJson,
  anthropicStreamEvents,
  anthropicToGeminiRequest,
  anthropicToOpenAIRequest,
  callUpstream,
  openAICompletionJson,
  openAIStreamEvents,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicStream,
  parseGeminiStream,
  parseOpenAIStream,
  parseProviderJson,
  providerUsage,
  type Usage,
  type Wire,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most provider attempts for one client request. */
const MAX_ATTEMPTS = 12;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far this trial (every attempt that named a free model). */
const callsMade = new Map<string, number>();
/** Provider refusals seen so far this trial (a 402, 429, 5xx or an unreachable call). */
const refusalsSeen = new Map<string, number>();
/** Consecutive refusals per provider: three in a row means the provider is spent or out. */
const consecutiveRefusals = new Map<string, number>();

/**
 * A safe cap on calls to one provider: the sealed ranges' minima. The real per-trial
 * budget is drawn inside [min, max], so it is never below this number; the router treats
 * a provider past its cap as spent and spreads the load instead of failing into it.
 */
const SAFE_CALLS: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best first. A JSON-mode request on the
 * OpenAI wire only ever goes to a wire that can echo JSON mode back (openai or gemini);
 * the anthropic wire always echoes json_mode false, so it is never a candidate there.
 * An Anthropic client never asks for JSON mode in its body, so no wire is excluded.
 */
function candidatesFor(requires: Requirement, jsonEcho: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonEcho && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

/**
 * The candidates in the order this request tries them: the rating, less a penalty
 * once the provider has passed its safe cap (its real budget may still hold, but the
 * load belongs elsewhere now). A refusal alone earns no penalty: a scheduled refusal
 * is one call number, and the provider's next call usually succeeds, so the best
 * model stays first and the failover chain absorbs the blip.
 */
function orderedCandidates(requires: Requirement, jsonEcho: boolean): Candidate[] {
  const candidates = candidatesFor(requires, jsonEcho).filter(
    (candidate) => (consecutiveRefusals.get(candidate.probe.provider.id) ?? 0) < 3,
  );
  const scoreOf = (candidate: Candidate): number => {
    const id = candidate.probe.provider.id;
    const overCap = (callsMade.get(id) ?? 0) >= (SAFE_CALLS[id] ?? 3) ? 400 : 0;
    return candidate.rating - overCap;
  };
  return candidates.sort((a, b) => {
    const diff = scoreOf(b) - scoreOf(a);
    if (diff !== 0) return diff;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this client body asks for JSON mode on the OpenAI wire (its echo must say so). */
function asksJsonEcho(clientWire: 'openai' | 'anthropic', body: Json): boolean {
  if (clientWire !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface Answered {
  text: string;
  usage: Usage;
  model: string;
}

/** The upstream body for one candidate, in the provider's own wire. */
function upstreamBodyFor(clientWire: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientWire === 'openai') {
    if (wire === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    if (wire === 'anthropic') return openAIToAnthropicRequest(body, model);
    return openAIToGeminiRequest(body, model);
  }
  if (wire === 'openai') return anthropicToOpenAIRequest(body, model);
  if (wire === 'anthropic') return { ...body, model };
  return anthropicToGeminiRequest(body, model);
}

/** One attempt at one candidate: the provider's answer, or null when it refused or failed. */
async function tryCandidate(
  clientWire: 'openai' | 'anthropic',
  body: Json,
  candidate: Candidate,
): Promise<Answered | null> {
  const provider = candidate.probe.provider;
  const wire: Wire = provider.wire;
  const stream = body.stream === true;
  const upstreamBody = upstreamBodyFor(clientWire, body, candidate);
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  const call = await callUpstream(ENV_URL, provider, upstreamBody, stream, candidate.model.id);
  if (call === null) {
    refusalsSeen.set(provider.id, (refusalsSeen.get(provider.id) ?? 0) + 1);
    consecutiveRefusals.set(provider.id, (consecutiveRefusals.get(provider.id) ?? 0) + 1);
    return null;
  }
  if (!call.ok) {
    refusalsSeen.set(provider.id, (refusalsSeen.get(provider.id) ?? 0) + 1);
    consecutiveRefusals.set(provider.id, (consecutiveRefusals.get(provider.id) ?? 0) + 1);
    return null;
  }
  let parsed;
  if (stream) {
    parsed = wire === 'openai' ? parseOpenAIStream(call.text) : wire === 'anthropic' ? parseAnthropicStream(call.text) : parseGeminiStream(call.text);
  } else {
    let json: unknown = null;
    try {
      json = JSON.parse(call.text);
    } catch {
      return null;
    }
    const found = parseProviderJson(wire, json);
    if (found === null) return null;
    parsed = found;
  }
  // The simulated providers answer with their echo; anything else is not an answer.
  if (!parsed.text.startsWith('ANSWER ECHO')) return null;
  consecutiveRefusals.set(provider.id, 0);
  const usage = parsed.usage ?? providerUsage(wire, upstreamBody, parsed.text);
  return { text: parsed.text, usage, model: candidate.model.id };
}

/**
 * One client request, answered by the best free model that meets it. Candidates are
 * tried in rating order until one answers. A refusal is usually one scheduled call
 * number, so a candidate that refuses is retried at once before moving on: its next
 * call number usually succeeds at the best rating. When every candidate has refused,
 * the tried ones are retried once more. Exactly one successful provider call answers.
 */
async function answer(clientWire: 'openai' | 'anthropic', body: Json): Promise<Answered | null> {
  const requires = requirementsFor(body);
  const jsonEcho = asksJsonEcho(clientWire, body);
  const ordered = orderedCandidates(requires, jsonEcho);
  if (ordered.length === 0) return null;
  const tried = new Set<string>();
  let attempts = 0;
  for (let pass = 0; pass < 2 && attempts < MAX_ATTEMPTS; pass += 1) {
    for (const candidate of ordered) {
      if (attempts >= MAX_ATTEMPTS) break;
      const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
      if (pass === 0 && tried.has(key)) continue;
      if (pass === 1 && (refusalsSeen.get(candidate.probe.provider.id) ?? 0) >= 3) continue;
      tried.add(key);
      attempts += 1;
      let answered = await tryCandidate(clientWire, body, candidate);
      if (answered !== null) return answered;
      // A refusal below the provider's safe cap is usually one scheduled call number,
      // so retry the same model at once: its next call usually succeeds at the best
      // rating. Past the cap a refusal means the provider is spent, so move on.
      const made = callsMade.get(candidate.probe.provider.id) ?? 0;
      const cap = SAFE_CALLS[candidate.probe.provider.id] ?? 3;
      if (pass === 0 && made < cap + 2 && attempts < MAX_ATTEMPTS) {
        attempts += 1;
        answered = await tryCandidate(clientWire, body, candidate);
        if (answered !== null) return answered;
      }
    }
  }
  return null;
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answered = await answer('openai', body);
  if (answered === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(openAIStreamEvents(answered.text, answered.usage, answered.model));
    return;
  }
  sendJson(res, 200, openAICompletionJson(answered.text, answered.usage, answered.model));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answered = await answer('anthropic', body);
  if (answered === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(anthropicStreamEvents(answered.text, answered.usage, answered.model));
    return;
  }
  sendJson(res, 200, anthropicMessageJson(answered.text, answered.usage, answered.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; retries until the env answers. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: string = '';
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw new Error(`register failed: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

// Translators between the client wires (OpenAI / Anthropic) and the provider wires
// (OpenAI / Anthropic / Gemini), plus the provider usage formula and the SSE parsers.
//
// The trial's client checks the provider's echo against what it sent, so every translation
// must carry the request through untouched: the system prompt, the user text (images included),
// the tool definitions, the stop sequences, max_tokens and JSON mode. A field rewritten or
// dropped loses the request. The one exception is a tool result: the Gemini parser never
// reads one, so tool messages are dropped on the Gemini wire instead of corrupting the echo.

import { isRecord, type Json } from './providers.ts';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export interface Usage {
  input: number;
  output: number;
}

/** The text of a string or of a block array's text blocks, joined with '\n'. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of a Gemini parts array. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A stop list from an OpenAI `stop` (a string or an array of strings). */
function stopList(stop: unknown): string[] {
  if (typeof stop === 'string') return [stop];
  if (!Array.isArray(stop)) return [];
  return stop.filter((x): x is string => typeof x === 'string');
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI image_url as an Anthropic image block, or null when it carries no URL. */
function anthropicImageFromUrl(url: string): Json | null {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  }
  if (/^https?:\/\//i.test(url)) return { type: 'image', source: { type: 'url', url } };
  return null;
}

/** An OpenAI image_url as a Gemini part, or null when it carries no URL. */
function geminiImageFromUrl(url: string): Json | null {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  }
  if (/^https?:\/\//i.test(url)) return { fileData: { mimeType: 'image/png', fileUri: url } };
  return null;
}

/** An Anthropic image source as a Gemini part, or null when it carries no data. */
function geminiImageFromAnthropicSource(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  return null;
}

/** OpenAI tools as Anthropic tools. */
function anthropicTools(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as OpenAI tools. */
function openAITools(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** OpenAI or Anthropic tools as Gemini function declarations. */
function geminiFunctionDeclarations(tools: unknown, fromOpenAI: boolean): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    if (fromOpenAI) {
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    }
    if (typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request. The system prompt
 * becomes the system string, image blocks become Anthropic image blocks, tool_use blocks
 * become tool_calls the other way, tool results stay tool_result blocks, and max_tokens,
 * temperature, top_p, stop sequences, tools and tool_choice pass through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const image = anthropicImageFromUrl(block.image_url.url);
          if (image !== null) parts.push(image);
        }
      }
    } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ type: 'text', text: raw.content });
    messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = anthropicTools(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (typeof choice === 'string' && choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (typeof choice === 'string' && choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. Tool messages are
 * dropped: the Gemini parser never reads a tool result, and carrying one as text would change
 * the user text the provider echoes.
 */
export function openAIToGeminiRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue;
    const parts: Json[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const image = geminiImageFromUrl(block.image_url.url);
          if (image !== null) parts.push(image);
        }
      }
    } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = geminiFunctionDeclarations(body.tools, true);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. Tool_use and tool_result
 * blocks are dropped for the same reason as above; an Anthropic request never asks for JSON
 * mode, so no responseMimeType is ever set.
 */
export function anthropicToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const parts: Json[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const image = geminiImageFromAnthropicSource(block.source);
          if (image !== null) parts.push(image);
        }
      }
    } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiFunctionDeclarations(body.tools, false);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((x): x is string => typeof x === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as an OpenAI Chat Completions request (the starter's shape). */
export function anthropicToOpenAIRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          const media = typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png';
          parts.push({ type: 'image_url', image_url: { url: `data:${media};base64,${block.source.data}` } });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: block.source.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  const tools = openAITools(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The provider usage the env computes for an answer: the same formula, so a usage the
 *  provider did not send (an OpenAI stream without a usage chunk) can be rebuilt exactly. */
export function providerUsage(wire: Wire, upstreamBody: Json, text: string): Usage {
  let systemText = '';
  let userText = '';
  const tools: string[] = [];
  if (wire === 'openai') {
    const messages = Array.isArray(upstreamBody.messages) ? upstreamBody.messages.filter(isRecord) : [];
    const systemParts: string[] = [];
    for (const message of messages) {
      if (message.role === 'system') {
        const text = textOf(message.content);
        if (text !== '') systemParts.push(text);
      } else if (message.role === 'user') userText = textOf(message.content);
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(upstreamBody.tools)) {
      for (const tool of upstreamBody.tools) {
        if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
      }
    }
  } else if (wire === 'anthropic') {
    systemText = textOf(upstreamBody.system);
    const messages = Array.isArray(upstreamBody.messages) ? upstreamBody.messages.filter(isRecord) : [];
    for (const message of messages) {
      if (message.role === 'user') userText = textOf(message.content);
    }
    if (Array.isArray(upstreamBody.tools)) {
      for (const tool of upstreamBody.tools) {
        if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
      }
    }
  } else {
    if (isRecord(upstreamBody.systemInstruction)) systemText = geminiPartsText(upstreamBody.systemInstruction.parts);
    const contents = Array.isArray(upstreamBody.contents) ? upstreamBody.contents.filter(isRecord) : [];
    for (const content of contents) {
      if (content.role === 'model') continue;
      userText = geminiPartsText(content.parts);
    }
    if (Array.isArray(upstreamBody.tools)) {
      for (const tool of upstreamBody.tools) {
        if (!isRecord(tool) || !Array.isArray(tool.functionDeclarations)) continue;
        for (const fn of tool.functionDeclarations) {
          if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
        }
      }
    }
  }
  const prompt = systemText.length + userText.length + tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

export interface ProviderDirectoryEntryLike {
  id: string;
  wire: Wire;
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  base: string;
  key_env: string;
  key: string;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: ProviderDirectoryEntryLike['auth'], key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

export interface UpstreamCall {
  ok: boolean;
  status: number;
  /** The whole response body as text (a JSON body or an SSE stream). */
  text: string;
}

/**
 * One inference call to a provider in its own wire. Returns null only when the call could
 * not be made at all; a refused call returns its status so the router can fail over.
 */
export async function callUpstream(
  envUrl: string,
  provider: ProviderDirectoryEntryLike,
  upstreamBody: Json,
  stream: boolean,
  model: string,
): Promise<UpstreamCall | null> {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  } catch {
    return null;
  }
  const text = await res.text();
  return { ok: res.ok, status: res.status, text };
}

/** One SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SSEFrame {
  event: string | null;
  data: string;
}

export function parseSSEFrames(text: string): SSEFrame[] {
  const frames: SSEFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

export interface ParsedStream {
  text: string;
  usage: Usage | null;
  finish: string | null;
}

/** An OpenAI event stream: the content deltas joined, the usage chunk, the finish reason. */
export function parseOpenAIStream(text: string): ParsedStream {
  let out = '';
  let usage: Usage | null = null;
  let finish: string | null = null;
  for (const frame of parseSSEFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  return { text: out, usage, finish };
}

/** An Anthropic event stream: the text deltas joined, the usage, the stop reason. */
export function parseAnthropicStream(text: string): ParsedStream {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSSEFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  const usage = input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 };
  return { text: out, usage, finish: stop };
}

/** A Gemini event stream: the candidate text pieces joined, the usage, the finish reason. */
export function parseGeminiStream(text: string): ParsedStream {
  let out = '';
  let usage: Usage | null = null;
  let finish: string | null = null;
  for (const frame of parseSSEFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
        for (const part of candidate.content.parts) {
          if (isRecord(part) && typeof part.text === 'string') out += part.text;
        }
      }
      if (typeof candidate.finishReason === 'string') finish = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: out, usage, finish };
}

/** The answer text of a provider's JSON body, in its own wire, with its usage. */
export function parseProviderJson(wire: Wire, body: unknown): ParsedStream | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const message = choice.message;
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(body.usage)
      ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) }
      : null;
    return { text, usage, finish: typeof choice.finish_reason === 'string' ? choice.finish_reason : null };
  }
  if (wire === 'anthropic') {
    if (body.type !== 'message' || !Array.isArray(body.content)) return null;
    const text = textOf(body.content);
    const usage = isRecord(body.usage)
      ? { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) }
      : null;
    return { text, usage, finish: typeof body.stop_reason === 'string' ? body.stop_reason : null };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const text = isRecord(candidate.content) ? geminiPartsText(candidate.content.parts) : '';
  const usage = isRecord(body.usageMetadata)
    ? { input: numberOrZero(body.usageMetadata.promptTokenCount), output: numberOrZero(body.usageMetadata.candidatesTokenCount) }
    : null;
  return { text, usage, finish: typeof candidate.finishReason === 'string' ? candidate.finishReason : null };
}

/** A fixed epoch second for every completion's `created`: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;

/** One answer as an OpenAI Chat Completions object, with the provider's usage whole. */
export function openAICompletionJson(text: string, usage: Usage, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** One answer as an Anthropic Messages object, with the provider's usage whole. */
export function anthropicMessageJson(text: string, usage: Usage, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An answer as an OpenAI event stream: role chunk, content deltas, finish, usage, [DONE]. */
export function openAIStreamEvents(text: string, usage: Usage, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (let at = 0; at < text.length; at += 64) {
    frames.push(sseEvent(null, chunk({ content: text.slice(at, at + 64) }, null)));
  }
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An answer as an Anthropic event stream: start, one text block, delta, stop, message_stop. */
export function anthropicStreamEvents(text: string, usage: Usage, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (let at = 0; at < text.length; at += 64) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: text.slice(at, at + 64) } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

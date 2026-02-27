import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatOllama } from '@langchain/ollama';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { StructuredToolInterface } from '@langchain/core/tools';
import { Runnable } from '@langchain/core/runnables';
import { z } from 'zod';
import { DEFAULT_SYSTEM_PROMPT } from '@/agent/prompts';

export const DEFAULT_PROVIDER = 'openai';
export const DEFAULT_MODEL = 'gpt-5.2';

// Fast model variants by provider for lightweight tasks like summarization
const FAST_MODELS: Record<string, string> = {
  openai: 'gpt-4.1',
  anthropic: 'claude-haiku-4-5',
  google: 'gemini-3-flash-preview',
  xai: 'grok-4-1-fast-reasoning',
};

/**
 * Gets the fast model variant for the given provider.
 * Falls back to the provided model if no fast variant is configured (e.g., Ollama).
 */
export function getFastModel(modelProvider: string, fallbackModel: string): string {
  return FAST_MODELS[modelProvider] ?? fallbackModel;
}

// Generic retry helper with exponential backoff
async function withRetry<T>(fn: () => Promise<T>, maxAttempts = 3): Promise<T> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (e) {
      if (attempt === maxAttempts - 1) throw e;
      await new Promise((r) => setTimeout(r, 500 * 2 ** attempt));
    }
  }
  throw new Error('Unreachable');
}

// Model provider configuration
interface ModelOpts {
  streaming: boolean;
  apiKeys?: Record<string, string>;
}

type ModelFactory = (name: string, opts: ModelOpts) => BaseChatModel;

function getApiKey(envVar: string, providerName: string, overrides?: Record<string, string>): string {
  const apiKey = overrides?.[envVar] || process.env[envVar];
  if (!apiKey) {
    throw new Error(`${envVar} not found in environment variables or client overrides`);
  }
  return apiKey;
}

const MODEL_PROVIDERS: Record<string, ModelFactory> = {
  'claude-': (name, opts) =>
    new ChatAnthropic({
      model: name,
      ...opts,
      apiKey: getApiKey('ANTHROPIC_API_KEY', 'Anthropic', opts.apiKeys),
    }),
  'gemini-': (name, opts) =>
    new ChatGoogleGenerativeAI({
      model: name,
      ...opts,
      apiKey: getApiKey('GOOGLE_API_KEY', 'Google', opts.apiKeys),
    }),
  'grok-': (name, opts) =>
    new ChatOpenAI({
      model: name,
      ...opts,
      apiKey: getApiKey('XAI_API_KEY', 'xAI', opts.apiKeys),
      configuration: {
        baseURL: 'https://api.x.ai/v1',
      },
    }),
  'openrouter/': (name, opts) =>
    new ChatOpenAI({
      model: name.replace(/^openrouter\//, ''),
      ...opts,
      apiKey: getApiKey('OPENROUTER_API_KEY', 'OpenRouter', opts.apiKeys),
      configuration: {
        baseURL: 'https://openrouter.ai/api/v1',
        defaultHeaders: {
          'HTTP-Referer': 'https://github.com/virattt/dexter',
          'X-Title': 'Dexter',
        },
      },
    }),
  'ollama:': (name, opts) =>
    new ChatOllama({
      model: name.replace(/^ollama:/, ''),
      ...opts,
      ...(process.env.OLLAMA_BASE_URL ? { baseUrl: process.env.OLLAMA_BASE_URL } : {}),
    }),
};

const DEFAULT_MODEL_FACTORY: ModelFactory = (name, opts) =>
  new ChatOpenAI({
    model: name,
    ...opts,
    apiKey: opts.apiKeys?.['OPENAI_API_KEY'] || process.env.OPENAI_API_KEY,
  });

export function getChatModel(
  modelName: string = DEFAULT_MODEL,
  streaming: boolean = false,
  apiKeys?: Record<string, string>
): BaseChatModel {
  const opts: ModelOpts = { streaming, apiKeys };
  const prefix = Object.keys(MODEL_PROVIDERS).find((p) => modelName.startsWith(p));
  const factory = prefix ? MODEL_PROVIDERS[prefix] : DEFAULT_MODEL_FACTORY;
  return factory(modelName, opts);
}

interface CallLlmOptions {
  model?: string;
  systemPrompt?: string;
  outputSchema?: z.ZodType<unknown>;
  tools?: StructuredToolInterface[];
  signal?: AbortSignal;
  apiKeys?: Record<string, string>;
}

export async function callLlm(prompt: string, options: CallLlmOptions = {}): Promise<unknown> {
  const { model = DEFAULT_MODEL, systemPrompt, outputSchema, tools, signal, apiKeys } = options;
  const finalSystemPrompt = systemPrompt || DEFAULT_SYSTEM_PROMPT;

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ['system', finalSystemPrompt],
    ['user', '{prompt}'],
  ]);

  const llm = getChatModel(model, false, apiKeys);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let runnable: Runnable<any, any> = llm;

  if (outputSchema) {
    runnable = llm.withStructuredOutput(outputSchema, { strict: false });
  } else if (tools && tools.length > 0 && llm.bindTools) {
    runnable = llm.bindTools(tools);
  }

  const chain = promptTemplate.pipe(runnable);

  console.log(`[LLM] Calling model: ${model} with tools: ${tools?.length || 0}`);
  const start = Date.now();
  try {
    const result = await withRetry(() => chain.invoke({ prompt }, signal ? { signal } : undefined));
    console.log(`[LLM] Result received in ${Date.now() - start}ms`);
    // If no outputSchema and no tools, extract content from AIMessage
    // When tools are provided, return the full AIMessage to preserve tool_calls
    if (!outputSchema && !tools && result && typeof result === 'object' && 'content' in result) {
      return (result as { content: string }).content;
    }
    return result;
  } catch (e) {
    console.error(`[LLM] Call failed after ${Date.now() - start}ms:`, e);
    throw e;
  }
}

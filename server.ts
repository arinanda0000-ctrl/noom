import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '10mb' }));

// Helper to get initialized GoogleGenAI instance
function getGenAI() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is missing.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Health check API
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

function formatContents(messages: any[]) {
  return (messages || []).map((m: any) => {
    const parts: any[] = [];
    if (m.attachments && Array.isArray(m.attachments)) {
      m.attachments.forEach((att: any) => {
        if (att.data && att.mimeType) {
          const base64Data = att.data.includes('base64,')
            ? att.data.split('base64,')[1]
            : att.data;
          parts.push({
            inlineData: {
              mimeType: att.mimeType,
              data: base64Data,
            },
          });
        }
      });
    }
    if (m.content) {
      parts.push({ text: m.content });
    }
    if (parts.length === 0) {
      parts.push({ text: '' });
    }
    return {
      role: m.role === 'user' ? 'user' : 'model',
      parts,
    };
  });
}

// Helper to get fallback model candidates
const FALLBACK_MODELS = [
  'gemini-3.6-flash',
  'gemini-3.1-flash-lite',
  'gemini-flash-latest',
];

function getModelCandidates(requestedModel?: string) {
  const primary = requestedModel || 'gemini-3.6-flash';
  const list = [primary, ...FALLBACK_MODELS.filter((m) => m !== primary)];
  return Array.from(new Set(list));
}

function sanitizeErrorMessage(err: any): string {
  if (!err) return 'An unknown error occurred.';
  const raw = typeof err === 'string' ? err : err.message || String(err);
  if (
    raw.includes('RESOURCE_EXHAUSTED') ||
    raw.includes('429') ||
    raw.includes('Quota exceeded') ||
    raw.includes('rate limit') ||
    raw.includes('quota') ||
    raw.includes('Too Many Requests')
  ) {
    return 'Gemini API quota or rate limit reached. Please wait a moment and try again.';
  }
  return raw;
}

async function generateContentWithFallback(ai: any, requestedModel: string, contents: any, config: any) {
  const modelsToTry = getModelCandidates(requestedModel);
  let lastError: any = null;

  for (const model of modelsToTry) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents,
        config,
      });
      return response;
    } catch (err: any) {
      console.warn(`Model candidate ${model} failed:`, err?.message || err);
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 300));
      continue;
    }
  }
  throw lastError;
}

async function* generateContentStreamWithFallback(ai: any, requestedModel: string, contents: any, config: any) {
  const modelsToTry = getModelCandidates(requestedModel);
  let lastError: any = null;

  for (const model of modelsToTry) {
    try {
      const responseStream = await ai.models.generateContentStream({
        model,
        contents,
        config,
      });
      let chunkCount = 0;
      for await (const chunk of responseStream) {
        chunkCount++;
        yield chunk;
      }
      if (chunkCount > 0) {
        return;
      }
    } catch (err: any) {
      console.warn(`Stream model candidate ${model} failed:`, err?.message || err);
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 300));
      continue;
    }
  }
  throw lastError;
}

// AI Chat API
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, systemInstruction, enableWebSearch, modelName } = req.body;

    const ai = getGenAI();
    const contents = formatContents(messages);

    const config: any = {};
    if (systemInstruction) {
      config.systemInstruction = systemInstruction;
    }
    if (enableWebSearch) {
      config.tools = [{ googleSearch: {} }];
    }

    const response = await generateContentWithFallback(ai, modelName || 'gemini-3.6-flash', contents, config);

    const replyText = response.text || 'No response generated.';
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];

    res.json({
      text: replyText,
      groundingChunks,
    });
  } catch (error: any) {
    console.error('Error in /api/chat:', error);
    res.status(500).json({ error: sanitizeErrorMessage(error) });
  }
});

// AI Chat Stream API (SSE)
app.post('/api/chat/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');

  try {
    const { messages, systemInstruction, enableWebSearch, modelName } = req.body;
    const ai = getGenAI();

    const contents = formatContents(messages);

    const config: any = {};
    if (systemInstruction) {
      config.systemInstruction = systemInstruction;
    }
    if (enableWebSearch) {
      config.tools = [{ googleSearch: {} }];
    }

    const responseStream = await generateContentStreamWithFallback(ai, modelName || 'gemini-3.6-flash', contents, config);

    for await (const chunk of responseStream) {
      const textChunk = chunk.text;
      if (textChunk) {
        res.write(`data: ${JSON.stringify({ text: textChunk })}\n\n`);
      }
    }

    res.write(`data: [DONE]\n\n`);
    res.end();
  } catch (error: any) {
    console.error('Error in /api/chat/stream:', error);
    res.write(`data: ${JSON.stringify({ error: sanitizeErrorMessage(error) })}\n\n`);
    res.end();
  }
});

// AI Writer API
app.post('/api/write', async (req, res) => {
  try {
    const {
      contentType, // 'blog' | 'email' | 'essay' | 'summary' | 'outline' | 'rephrase' | 'custom'
      prompt,
      tone,
      length,
      targetAudience,
      formatStyle,
    } = req.body;

    const ai = getGenAI();

    let systemInstruction = `You are NomAI Writer, a world-class professional AI writing assistant. Produce high-quality, beautifully structured content without fluff, AI clichés, or repetitive filler. Keep styling clean, using Markdown headers, lists, and bold callouts where appropriate.`;

    let userPrompt = `Content Task: ${contentType || 'Writing'}\nTopic/Prompt: ${prompt}\n`;

    if (tone) userPrompt += `Tone: ${tone}\n`;
    if (length) userPrompt += `Target Length: ${length}\n`;
    if (targetAudience) userPrompt += `Target Audience: ${targetAudience}\n`;
    if (formatStyle) userPrompt += `Format Preference: ${formatStyle}\n`;

    if (contentType === 'summary') {
      userPrompt += `\nPlease summarize the provided text accurately, capturing main takeaways, bulleted key points, and an executive overview.`;
    } else if (contentType === 'rephrase') {
      userPrompt += `\nPlease rephrase and polish the text to enhance clarity, flow, vocabulary, and conciseness while preserving original meaning.`;
    } else if (contentType === 'outline') {
      userPrompt += `\nPlease create a comprehensive, organized outline with clear main topics, subtopics, and bullet points.`;
    }

    const response = await generateContentWithFallback(
      ai,
      'gemini-3.6-flash',
      userPrompt,
      { systemInstruction }
    );

    res.json({ text: response.text || '' });
  } catch (error: any) {
    console.error('Error in /api/write:', error);
    res.status(500).json({ error: sanitizeErrorMessage(error) });
  }
});

// AI Translator API
app.post('/api/translate', async (req, res) => {
  try {
    const { text, sourceLang, targetLang, formality, includePronunciation } = req.body;

    const ai = getGenAI();

    const systemInstruction = `You are NomAI Translator, an expert linguistic AI model. Translate accurately, preserving nuance, idioms, and context. Do NOT add unnecessary commentary unless requested. Always output clean Markdown structure if formatting exists.`;

    const prompt = `Translate the following text from ${sourceLang || 'Auto-Detect'} to ${targetLang || 'English'}.
Formality Level: ${formality || 'Neutral/Standard'}
${includePronunciation ? 'Please also include a phonetic pronunciation/transliteration guide under the translation.' : ''}

Source Text:
"""
${text}
"""`;

    const response = await generateContentWithFallback(
      ai,
      'gemini-3.6-flash',
      prompt,
      { systemInstruction }
    );

    res.json({ text: response.text || '' });
  } catch (error: any) {
    console.error('Error in /api/translate:', error);
    res.status(500).json({ error: sanitizeErrorMessage(error) });
  }
});

// Vite middleware for development vs static production serving
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`NomAI Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();

// سرور واسط — روی Render دیپلوی می‌شه
// دو منبع رو جستجو می‌کنه: کتاب خلاصه‌ی فارسی (chunks_fa.json) و رفرنس کامل انگلیسی (chunks_en.json)
// چون سؤال فارسیه ولی رفرنس اصلی انگلیسیه، اول سؤال رو با Groq به انگلیسی ترجمه می‌کنیم
// تا بشه توی متن انگلیسی درست جستجو کرد، بعد جواب نهایی رو فارسی می‌سازیم

const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({ limit: '2mb' }));

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// --- بارگذاری منابع ---
const CHUNKS_FA = JSON.parse(fs.readFileSync(path.join(__dirname, 'chunks_fa.json'), 'utf-8'));
const CHUNKS_EN = JSON.parse(fs.readFileSync(path.join(__dirname, 'chunks_en.json'), 'utf-8'));
console.log(`بارگذاری شد: ${CHUNKS_FA.length} تیکه فارسی، ${CHUNKS_EN.length} تیکه انگلیسی`);

// --- ابزار جستجوی TF-IDF ساده، برای هر زبان جدا ---

function normalizeFa(t) {
  return t
    .replace(/[\u064B-\u065F\u0670\u06D6-\u06ED]/g, '')
    .replace(/[إأآا]/g, 'ا')
    .replace(/ي/g, 'ی')
    .replace(/ك/g, 'ک')
    .replace(/ة/g, 'ه')
    .replace(/[\u200c\u200f\u202b\u202c]/g, ' ')
    .toLowerCase();
}
const STOPWORDS_FA = new Set(['و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'است', 'برای', 'آن', 'یک', 'های', 'هم', 'تا', 'یا', 'هر', 'می', 'شود', 'شده', 'ها', 'نیز', 'بر', 'اگر', 'چه', 'ولی', 'اما', 'چون', 'پس', 'بین', 'روی', 'زیر', 'بالا']);
function tokenizeFa(t) {
  return normalizeFa(t).split(/[^ا-یa-z0-9]+/).filter(w => w.length > 1 && !STOPWORDS_FA.has(w));
}

const STOPWORDS_EN = new Set(['the', 'and', 'of', 'to', 'in', 'a', 'is', 'for', 'with', 'as', 'are', 'by', 'that', 'this', 'be', 'or', 'an', 'on', 'from', 'it', 'can', 'may', 'these', 'which', 'has', 'have', 'not', 'also', 'such', '其', 'was', 'were']);
function tokenizeEn(t) {
  return t.toLowerCase().split(/[^a-z0-9]+/).filter(w => w.length > 1 && !STOPWORDS_EN.has(w));
}

function buildIndex(chunks, tokenizer) {
  const tokensList = chunks.map(c => tokenizer(c));
  const df = {};
  tokensList.forEach(tokens => {
    new Set(tokens).forEach(w => { df[w] = (df[w] || 0) + 1; });
  });
  return { chunks, tokensList, df, N: chunks.length };
}

const INDEX_FA = buildIndex(CHUNKS_FA, tokenizeFa);
const INDEX_EN = buildIndex(CHUNKS_EN, tokenizeEn);
console.log('پایگاه‌های جستجو آماده شدن.');

function search(index, queryTokens, topK) {
  const scored = index.chunks.map((text, i) => {
    const tokens = index.tokensList[i];
    const tf = {};
    tokens.forEach(w => { tf[w] = (tf[w] || 0) + 1; });
    let score = 0;
    queryTokens.forEach(qw => {
      if (tf[qw]) {
        const idf = Math.log((index.N + 1) / ((index.df[qw] || 0) + 1)) + 1;
        score += tf[qw] * idf;
      }
    });
    return { text, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.filter(s => s.score > 0).slice(0, topK);
}

// --- تماس با Groq ---

async function callGroq(messages, temperature = 0.15) {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
    },
    body: JSON.stringify({ model: 'openai/gpt-oss-120b', temperature, messages }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`خطای Groq: ${errText}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? '';
}

async function translateToEnglishQuery(question) {
  const content = await callGroq([
    { role: 'system', content: 'Translate the nutrition-related question into a short English search query of 3-8 keywords only. No explanation, no extra text, just the keywords.' },
    { role: 'user', content: question },
  ], 0.0);
  return content.trim();
}

const SYSTEM_PROMPT =
  'تو یه متخصص تغذیه با تحلیل عمیق هستی که به سؤالات کاربر جواب کامل و مستدل می‌ده. ' +
  'بخش‌های متنی که بهت داده می‌شه فقط پایه‌ی دانش توئه — نه چیزی که باید بهش اشاره یا استناد کنی. ' +
  'قوانین: ' +
  '1) هیچ‌وقت نگو «طبق کتاب»، «توی این بخش»، «طبق توضیحات صفحه‌ی قبل»، «توی متن اومده» یا هر اشاره‌ای به اینکه داری از یه سند خاص می‌خونی — کاربر نباید بفهمه پشت جواب یه رفرنس هست، فقط باید حس کنه با یه متخصص واقعی حرف می‌زنه. ' +
  '2) هیچ‌وقت جمله یا پاراگراف رو عیناً کپی نکن — همیشه مطلب رو کامل هضم کن، با کلمات خودت و در قالب یه توضیح روان و طبیعی بازش کن، انگار خودت این دانش رو در حافظه داری. ' +
  '3) فقط به نکات جزئی و پراکنده بسنده نکن — نکات مرتبط رو با هم ترکیب کن، ارتباطشون رو نشون بده، و یه تحلیل کامل و کاربردی بده (چرا این‌طوره، چه اهمیتی داره، چه نکته‌ی عملی‌ای ازش درمیاد). ' +
  '4) اگه دانشی که در اختیارت گذاشته شده کامل نیست، از دانش عمومی معتبر تغذیه‌ای خودت برای کامل کردن جواب استفاده کن — بدون اینکه بگی کدوم بخش از کجا اومده؛ جواب باید یکپارچه و طبیعی باشه. ' +
  '5) هیچ‌وقت اطلاعات نادرست یا من‌درآوردی نساز؛ اگه از چیزی مطمئن نیستی، صادقانه بگو مطمئن نیستم، بدون اینکه به منبع اشاره کنی. ' +
  '6) همیشه جواب رو به فارسی روان، دقیق و با عمق تحلیلی بنویس. ' +
  '7) اگه کاربر پرسید این برنامه/اپلیکیشن رو کی ساخته یا سازنده‌ش کیه، دقیقاً همین رو جواب بده و به هیچ‌چیز دیگه‌ای اشاره نکن: «این برنامه توسط سجاد رهنما دانشجوی رشته تغذیه ساخته شده است.»';

app.get('/', (req, res) => {
  res.send(`سرور واسط تغذیه فعاله. (${CHUNKS_FA.length} تیکه فارسی، ${CHUNKS_EN.length} تیکه انگلیسی)`);
});

app.post('/', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: 'question لازمه' });
    }

    // اگه سؤال درباره‌ی سازنده بود، نیازی به جستجو نیست
    const creatorPatterns = /سازنده|کی ساخته|چه کسی ساخت|طراح این برنامه/;
    let englishQuery = '';
    if (!creatorPatterns.test(question)) {
      englishQuery = await translateToEnglishQuery(question);
    }

    const topEn = englishQuery ? search(INDEX_EN, tokenizeEn(englishQuery), 6) : [];
    const topFa = search(INDEX_FA, tokenizeFa(question), 3);

    const contextParts = [];
    topFa.forEach(r => contextParts.push(r.text));
    topEn.forEach(r => contextParts.push(r.text));

    const context = contextParts.length > 0
      ? contextParts.join('\n\n')
      : '(دانش خاصی برای این سؤال در دسترس نبود، از دانش عمومی معتبر تغذیه استفاده کن)';

    const answer = await callGroq([
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: `دانش پس‌زمینه (فقط برای استفاده‌ی داخلی خودت، هیچ‌وقت بهش اشاره نکن):\n${context}\n\nسؤال کاربر: ${question}` },
    ], 0.3);

    res.json({
      answer,
      sources: [
        ...topFa.map(r => ({ source: 'کتاب فارسی کراوس', text: r.text })),
        ...topEn.map(r => ({ source: "Krause's Food & Nutrition Care Process", text: r.text })),
      ],
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`روی پورت ${PORT} در حال اجراست`));

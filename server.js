// سرور واسط ساده — این رو روی Render.com دیپلوی کن
// چون Render برخلاف Cloudflare Workers یه سرور با مکان ثابت بهت می‌ده
// (نه یه چیزی که بسته به مکان کاربر جابه‌جا بشه)، مشکل قبلی دیگه پیش نمیاد

const express = require('express');
const app = express();
app.use(express.json({ limit: '2mb' }));

// اجازه‌ی درخواست از هر جایی
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

app.get('/', (req, res) => {
  res.send('سرور واسط تغذیه فعاله.');
});

app.post('/', async (req, res) => {
  try {
    const { context, question } = req.body;
    if (!context || !question) {
      return res.status(400).json({ error: 'context و question لازمه' });
    }

    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        temperature: 0.15,
        messages: [
          {
            role: 'system',
            content:
              'تو یه دستیار متخصص تغذیه هستی. فقط و فقط بر اساس بخش‌های متنی که کاربر از کتاب می‌ده جواب بده. ' +
              'قوانین سخت‌گیرانه: 1) هیچ اطلاعاتی از دانش عمومی یا حدس خودت اضافه نکن، فقط از متن داده‌شده استفاده کن. ' +
              '2) اگه جواب دقیق سؤال توی متن نیست، صریح بگو: «توی این بخش از کتاب پاسخ این سؤال پیدا نشد.» ' +
              '3) اگه بخشی از متن نامربوط بود، نادیده‌اش بگیر. ' +
              '4) جواب رو کوتاه، دقیق و به فارسی روان بنویس؛ از حدسیات پرهیز کن.',
          },
          { role: 'user', content: `بخش‌های کتاب:\n${context}\n\nسؤال: ${question}\n\nفقط بر اساس متن بالا جواب بده.` },
        ],
      }),
    });

    if (!groqRes.ok) {
      const errText = await groqRes.text();
      return res.status(502).json({ error: `خطای Groq: ${errText}` });
    }

    const data = await groqRes.json();
    const answer = data.choices?.[0]?.message?.content ?? 'جوابی برنگشت.';
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`روی پورت ${PORT} در حال اجراست`));

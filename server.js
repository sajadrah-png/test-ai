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
              'تو یه دستیار متخصص تغذیه هستی. اولویت اول همیشه بخش‌های متنی کتاب هست که کاربر می‌ده. ' +
              'قوانین: 1) اگه جواب سؤال توی متن کتاب هست، فقط و دقیقاً بر همون اساس جواب بده، بدون افزودن چیز دیگه. ' +
              '2) اگه جواب توی متن کتاب نیست یا ناقصه، می‌تونی از دانش عمومی معتبر تغذیه‌ای خودت کمک بگیری، ولی این بخش رو صریح و واضح مشخص کن — مثلاً با شروع جمله‌ای مثل «این بخش از کتاب نیست، ولی طبق دانش عمومی تغذیه:». ' +
              '3) هیچ‌وقت این دو منبع رو با هم قاطی نکن که کاربر نفهمه کدوم از کتابه و کدوم نیست. ' +
              '4) هیچ‌وقت اطلاعات نادرست یا غیرمطمئن نساز؛ اگه از چیزی مطمئن نیستی، صریح بگو مطمئن نیستم. ' +
              '5) جواب رو دقیق، مفید و به فارسی روان بنویس. ' +
              '6) اگه کاربر پرسید این برنامه/اپلیکیشن رو کی ساخته یا سازنده‌ش کیه، دقیقاً همین رو جواب بده و به هیچ‌چیز دیگه‌ای اشاره نکن: «این برنامه توسط سجاد رهنما دانشجوی رشته تغذیه ساخته شده است.»',
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

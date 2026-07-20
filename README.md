# ASG Transformer — Professional Edition

إعادة بناء احترافية لمشروع ASG لتصنيف معلومات التهديدات السيبرانية وتوليد سيناريوهات هجوم مرتبة باستخدام Transformer دلالي.

## ما الذي تغير؟

- استبدال TF-IDF/SVM وملفات النماذج القديمة بطبقة Transformer قابلة للتدريب.
- إزالة المسارات الثابتة والاعتماد على إعدادات بيئية.
- فصل واضح بين البيانات، النموذج، منطق السيناريو، التدريب، والـAPI.
- منع القوائم المشتركة mutable defaults وكتل `except` الصامتة.
- توليد السيناريو باستخدام Beam Search يجمع التشابه الدلالي وانتقال التقنيات.
- API حديث باستخدام FastAPI مع مخططات Pydantic وتوثيق Swagger تلقائي.

## المعمارية

```text
Input Description
      │
      ▼
Transformer Bi-Encoder
      │
      ├── Technique Retrieval (78)
      ├── Software Retrieval (19)
      └── Threat Group Retrieval (13)
      │
      ▼
Tactic-aware Beam Search
Semantic Score + Transition Score
      │
      ▼
Ordered Cyber Scenario + Confidence
```

## التشغيل

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"
cp .env.example .env
uvicorn asg_transformer.api.main:app --reload
```

Swagger: `http://localhost:8000/docs`

## تدريب الـTransformer

```bash
python -m asg_transformer.training.train_encoder \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 10 \
  --batch-size 16
```

بعد التدريب سيُحفظ النموذج في `models/asg-encoder`، وستستخدمه الخدمة تلقائيًا.

## أمثلة API

```bash
curl -X POST http://localhost:8000/v1/classify/technique \
  -H "Content-Type: application/json" \
  -d '{"text":"An adversary changes PLC logic and suppresses alarms","top_k":5}'

 curl -X POST http://localhost:8000/v1/scenarios/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Remote access to an industrial controller followed by unsafe process manipulation","max_steps":6,"beam_width":5}'
```

## ملاحظة علمية

البيانات الأصلية تحتوي غالبًا على وصف واحد لكل فئة؛ لذلك يعتمد الإصدار الحالي على fine-tuning دلالي contrastive بدل classifier مغلق. لتحسين الدقة أكثر، أضف أمثلة واقعية متعددة لكل تقنية ومجموعة وبرمجية، ثم درّب النموذج مجددًا وقِس Recall@K وMRR وNDCG.

# -*- coding: utf-8 -*-
"""
================================================================================
 智慧警政 FB 高互動內容 AI 生成與驗證雙引擎系統  v2.8 (Gemini 圖像／Veo 影片後端版)
--------------------------------------------------------------------------------
 v2.8.1 修正重點（提示詞與影片一致性）：
   [7] Veo 專用提示詞組裝 _build_veo_prompt()：分鏡放最前面、制服/風格規範壓縮成短句、所有「嚴禁」
       改放 negative_prompt，並附上 GPT-4o 同步產出的英文分鏡摘要 (video_prompt_en) 供影片模型對齊；
       影片預設時長改為 VIDEO_DURATION_SEC=8 秒（4 秒塞不下四個鏡頭）；勾選「影片沿用圖片場景」時
       改用「圖片場景 + 動態指示」作為提示詞（Image-to-Video 一定會延續首幀畫面，不能再餵不同故事）。
       日誌會印出實際送給 Veo 的提示詞前 300 字，方便比對。

 v2.8 修正重點（相對於 v2.7）：
   [6] 多模態渲染層改接 Google Gemini API：圖像由 gpt-image-2 改為 Gemini 3.1 Flash Image，
       影片由 sora-2（OpenAI 已公告 2026-09-24 自 API 移除）改為 Veo 3.1。文案 GPT-4o、BERT 評估、
       RAG 與 XAI 模組完全不動；新增 IMAGE_BACKEND / VIDEO_BACKEND 常數與統一入口
       generate_image() / generate_video()，日後更換供應商只需改常數並註冊新函式。
       原 generate_image_gpt_image_2() / generate_video_sora() 保留，可隨時切回 "openai" / "sora"。

 v2.7 修正重點（相對於 v2.6）：
   [1] 視覺多元化：新增「視覺主題輪替池 (VISUAL_THEMES)」，每個候選方案指派不同主題，
       並移除 System Prompt / 反饋引擎中「積極融入警犬」的硬性導向，解決每次都出現警犬的問題。
   [2] 臺灣警察制服鎖定：新增 TW_POLICE_RULE，明確描述中華民國警察制服細節，並負面排除
       中國大陸公安制服；同時注入 GPT-4o 提示詞、gpt-image-2 與 sora-2 的最終 prompt。
   [3] XAI 顏色顛倒修正：Gradio HighlightedText 數值模式預設「正值=紅、負值=紫」，與標題
       「綠=拉高 / 紅=拖累」相反。改用分類標籤 + color_map，顏色與下方正/負向詞彙完全一致。
   [5] 圖片與影片分離為兩個故事：每個候選分別指派「圖片主題」與「影片主題」，System Prompt 要求
       地點/人物/時段/標語皆不同，影片以 4 鏡頭分鏡撰寫；影片預設不再以圖片作為首幀參考。
   [4] 疊代輪數控制：新增「最少疊代輪數」與「強制跑完所有輪數」選項；達標即停止不再是唯一行為，
       並於日誌輸出每輪成績總表。另將候選生成改為真正的多執行緒平行呼叫，縮短等待時間。

 執行指令 (Google Colab / 本地環境)：
   !pip install -q "gradio>=4.44" plotly jieba openai google-genai transformers torch pandas openpyxl scikit-learn
   %run police_fb_agent_v2_8.py
================================================================================
"""

import os
import re
import json
import time
import base64
import random
import inspect
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from openai import OpenAI
try:
    from google import genai                      # Google Gen AI SDK（pip install google-genai）
    from google.genai import types as gtypes
    HAS_GENAI = True
except ImportError:
    genai, gtypes, HAS_GENAI = None, None, False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import gradio as gr
import plotly.graph_objects as go

# 中文斷詞工具（若無 jieba 則自動啟用滑動視窗字元降級演算法）
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    jieba = None
    HAS_JIEBA = False

# =========================================================
# 1. 環境與 API Key 設定
# =========================================================
REAL_OPENAI_KEY = ""  # 👈 請填入您的 OpenAI Key（留空則依序改用 Colab Secrets / 環境變數 / 互動輸入）

def _resolve_openai_key() -> str:
    """依序嘗試四種來源取得 API Key，避免空字串覆蓋環境變數導致 Missing credentials"""
    # ① 程式內直接填寫
    if REAL_OPENAI_KEY and REAL_OPENAI_KEY.strip():
        return REAL_OPENAI_KEY.strip()
    # ② Colab 左側「🔑 Secrets」面板（名稱：OPENAI_API_KEY，記得開啟「筆記本存取權」）
    try:
        from google.colab import userdata  # type: ignore
        k = userdata.get("OPENAI_API_KEY")
        if k and k.strip():
            print("🔑 已從 Colab Secrets 讀取 OPENAI_API_KEY")
            return k.strip()
    except Exception:
        pass
    # ③ 既有環境變數
    k = os.environ.get("OPENAI_API_KEY", "")
    if k and k.strip():
        return k.strip()
    # ④ 互動式輸入（輸入時不顯示）
    try:
        from getpass import getpass
        k = getpass("🔑 請貼上您的 OpenAI API Key（sk-...），輸入後按 Enter：")
        if k and k.strip():
            return k.strip()
    except Exception:
        pass
    raise RuntimeError(
        "找不到 OpenAI API Key。請在程式第 56 行填入 REAL_OPENAI_KEY，"
        "或在 Colab Secrets 新增 OPENAI_API_KEY，或執行前設定環境變數 OPENAI_API_KEY。"
    )

_OPENAI_KEY = _resolve_openai_key()
os.environ["OPENAI_API_KEY"] = _OPENAI_KEY
client = OpenAI(api_key=_OPENAI_KEY)
print(f"✅ 已成功載入 OpenAI Client！（Key 末四碼：…{_OPENAI_KEY[-4:]}）")

# ---- [v2.8] Google Gemini 金鑰與 Client（負責圖像與影片渲染；文案仍由 OpenAI 端負責） ----
REAL_GOOGLE_KEY = ""  # 👈 請填入您的 Gemini API Key（留空則依序改用 Colab Secrets / 環境變數；可於 Google AI Studio 免費申請）

def _resolve_google_key() -> str:
    """與 OpenAI 相同的多來源取鑰策略；找不到時回傳空字串（僅影響圖片／影片渲染，不影響文案與評估）"""
    if REAL_GOOGLE_KEY and REAL_GOOGLE_KEY.strip():
        return REAL_GOOGLE_KEY.strip()
    try:
        from google.colab import userdata  # type: ignore
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            k = userdata.get(name)
            if k and k.strip():
                print(f"🔑 已從 Colab Secrets 讀取 {name}")
                return k.strip()
    except Exception:
        pass
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        k = os.environ.get(name, "")
        if k and k.strip():
            return k.strip()
    return ""

_GOOGLE_KEY = _resolve_google_key()
gclient = None
if HAS_GENAI and _GOOGLE_KEY:
    os.environ["GEMINI_API_KEY"] = _GOOGLE_KEY
    try:
        gclient = genai.Client(api_key=_GOOGLE_KEY)
    except Exception:
        gclient = genai.Client()               # 退回由環境變數尋找金鑰
    print(f"✅ 已成功載入 Google Gemini Client！（Key 末四碼：…{_GOOGLE_KEY[-4:]}）")
else:
    print("⚠️ 未載入 Google Gemini Client（缺少 google-genai 套件或金鑰）：圖片／影片渲染將不可用，文案生成與評估不受影響。")

# ---- [v2.8] 多模態渲染後端設定：更換供應商只需改這裡 ----
IMAGE_BACKEND = "gemini"                  # "gemini"＝Gemini 3.1 Flash Image；"openai"＝沿用 gpt-image-2
VIDEO_BACKEND = "veo"                     # "veo"＝Veo 3.1；"sora"＝舊版 sora-2（2026-09-24 起失效）
IMAGE_MODEL = "gemini-3.1-flash-image"    # 高品質構圖可改 "gemini-3-pro-image"
VEO_MODEL = "veo-3.1-generate-preview"    # 可改 Fast / Lite 變體或後繼版本（依 Google 官方模型清單）
VIDEO_DURATION_SEC = 8                    # Veo 3.1 支援 4/6/8 秒；四鏡頭分鏡建議 8 秒，想省費用可改 4
VEO_ADD_EN_HINT = True                    # 在提示詞末尾附英文分鏡摘要，提升影片模型對劇情的遵循度
IMAGE_MODEL_LABEL = IMAGE_MODEL if IMAGE_BACKEND == "gemini" else "gpt-image-2"
VIDEO_MODEL_LABEL = VEO_MODEL if VIDEO_BACKEND == "veo" else "sora-2"

MODEL_PATH = "/content/drive/MyDrive/投稿資料/各縣市警察局臉書貼文/六都/全50分佈_等量劃分資料集/best_checkpoint"
DATASET_PATH = "/content/drive/MyDrive/投稿資料/各縣市警察局臉書貼文/六都/六都警察局貼文資料_已移除發布時間-2.xlsx"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 2. 載入 BERT 評估模型與建置全域 RAG 知識庫
# =========================================================
print("🔄 正在載入微調後的 BERT 分類模型...")
tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese")

try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("✅ 成功載入本地微調 BERT 模型！")
except Exception as e:
    print(f"⚠️ 載入指定路徑失敗 ({e})，使用 ckiplab/bert-base-chinese 作為基礎模型")
    model = AutoModelForSequenceClassification.from_pretrained("ckiplab/bert-base-chinese", num_labels=2)

model.eval()
model.to(device)

print("📚 正在建立 RAG 爆款貼文知識庫...")
try:
    df_all = pd.read_excel(DATASET_PATH) if DATASET_PATH.endswith(".xlsx") else pd.read_csv(DATASET_PATH)
    if "label_extreme_25" in df_all.columns:
        df_high_engagement = df_all[df_all["label_extreme_25"] == 1].copy()
    elif "label_50" in df_all.columns:
        df_high_engagement = df_all[df_all["label_50"] == 1].copy()
    else:
        df_high_engagement = df_all.copy()
    df_high_engagement["Text_Clean"] = df_high_engagement["Text"].fillna("").astype(str)
except Exception as e:
    print(f"⚠️ 讀取歷史資料集失敗 ({e})，將採用預設生成模式。")
    df_high_engagement = pd.DataFrame(columns=["Text_Clean", "media_description"])

_RAG_TEXTS = df_high_engagement["Text_Clean"].tolist()
if _RAG_TEXTS:
    _RAG_VECTORIZER = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    _RAG_MATRIX = _RAG_VECTORIZER.fit_transform(_RAG_TEXTS)
    print(f"✅ RAG 知識庫預適配完成，共收錄 {len(_RAG_TEXTS)} 篇歷史爆款貼文！")
else:
    _RAG_VECTORIZER, _RAG_MATRIX = None, None

# =========================================================
# 3. RAG 檢索模組
# =========================================================
def retrieve_top_k_examples(topic: str, k: int = 2) -> str:
    """使用預適配 TF-IDF 矩陣極速檢索相似度最高之爆款範例"""
    if _RAG_MATRIX is None:
        return "（目前無歷史檢索語料，依預設社群行銷策略生成）"

    topic_vec = _RAG_VECTORIZER.transform([topic])
    similarities = cosine_similarity(topic_vec, _RAG_MATRIX).flatten()
    top_indices = similarities.argsort()[-k:][::-1]

    rag_context = ""
    for rank, idx in enumerate(top_indices, 1):
        row = df_high_engagement.iloc[idx]
        rag_context += f"\n--- [歷史爆款範例 {rank}] ---\n"
        rag_context += f"貼文內文：{row['Text_Clean'][:150]}...\n"
        rag_context += f"視覺描述：{str(row.get('media_description', '無'))[:100]}\n"
    return rag_context

# =========================================================
# 4. BERT 批次評估模組
# =========================================================
def build_bert_input(draft: dict) -> str:
    return (
        f"[貼文時間]{draft.get('publish_time', '')} "
        f"[貼文內文] {draft.get('text', '')} "
        f"[圖片特徵] {draft.get('image_desc', '')} "
        f"[影片動態] {draft.get('video_desc', '')}"
    )

@torch.no_grad()
def evaluate_batch(texts: list, batch_size: int = 16) -> list:
    if not texts:
        return []
    probs_all = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", max_length=384, padding="max_length", truncation=True
        ).to(device)
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[:, 1]
        probs_all.extend(probs.detach().cpu().tolist())
    return probs_all

def evaluate_post_quality(bert_input_text: str) -> float:
    return evaluate_batch([bert_input_text])[0]

# =========================================================
# 5. 可解釋性 AI (XAI)：反事實遮蔽法 (Occlusion Sensitivity)
# =========================================================
_STOPWORDS = set(
    "的 了 是 在 和 與 及 也 就 都 而 並 有 為 對 把 被 讓 使 給 從 到 於 之 其 這 那 "
    "我們 你們 他們 一個 可以 因為 所以 但是 如果 以及 或是 請 喔 唷 啦 呢 嗎".split()
)
_PUNCT_RE = re.compile(r"^[\W_、。，！？；：「」『』（）\s\d]+$", re.UNICODE)

def _segment(text: str) -> list:
    if HAS_JIEBA:
        return [w for w in jieba.cut(text) if w.strip()]
    tokens, buf = [], re.split(r"[，。！？、；：\s\n（）「」]+", text)
    for seg in buf:
        if not seg:
            continue
        if len(seg) <= 3:
            tokens.append(seg)
        else:
            for n in (3, 2):
                for i in range(0, len(seg) - n + 1, n):
                    tokens.append(seg[i : i + n])
    return tokens

# ---- [修正 3] HighlightedText 改用「分類標籤」模式 --------------------------
# Gradio HighlightedText 若傳入 float 分數，前端固定以「正值=紅色、負值=藍紫色」著色，
# 無法自訂，導致與標題「綠=拉高 / 紅=拖累」相反。改為分類標籤 + color_map 即可完全掌控顏色。
XAI_COLOR_MAP = {
    "強正向": "#1E8449",   # 深綠：顯著拉高爆款機率
    "正向":   "#A9DFBF",   # 淺綠：輕微拉高
    "負向":   "#F5B7B1",   # 淺紅：輕微拖累
    "強負向": "#C0392B",   # 深紅：顯著拖累
}

def _bucket(norm_val: float):
    """將標準化貢獻值 (-1~1) 轉為分類標籤；接近 0 者不著色 (None)"""
    if norm_val >= 0.5:
        return "強正向"
    if norm_val >= 0.1:
        return "正向"
    if norm_val <= -0.5:
        return "強負向"
    if norm_val <= -0.1:
        return "負向"
    return None

def explain_keywords(draft: dict, top_k: int = 10, max_candidates: int = 50) -> dict:
    base_text = build_bert_input(draft)
    base_prob = evaluate_post_quality(base_text)

    sections = {
        "內文": draft.get("text", ""),
        "圖片": draft.get("image_desc", ""),
        "影片": draft.get("video_desc", ""),
    }

    candidates = {}
    for sec_name, sec_text in sections.items():
        for w in _segment(sec_text):
            w = w.strip()
            if len(w) < 2 or w in _STOPWORDS or _PUNCT_RE.match(w):
                continue
            candidates.setdefault(w, sec_name)

    words = list(candidates.keys())[:max_candidates]
    if not words:
        return {"base_prob": base_prob, "items": [], "positives": [], "negatives": [], "highlight": []}

    variants = [base_text.replace(w, "") for w in words]
    variant_probs = evaluate_batch(variants)

    items = []
    for w, p in zip(words, variant_probs):
        items.append({"word": w, "section": candidates[w], "contribution": base_prob - p, "prob_without": p})
    items.sort(key=lambda x: x["contribution"], reverse=True)

    score_map = {it["word"]: it["contribution"] for it in items}
    max_abs = max((abs(v) for v in score_map.values()), default=1e-6) or 1e-6
    highlight = []
    for w in _segment(sections["內文"]):
        val = score_map.get(w.strip(), 0.0) / max_abs
        highlight.append((w, _bucket(val)))          # ← 分類標籤，而非 float

    positives = [it for it in items if it["contribution"] > 0][:top_k]
    negatives = [it for it in items if it["contribution"] < 0][-top_k:][::-1]

    return {"base_prob": base_prob, "items": items, "positives": positives, "negatives": negatives, "highlight": highlight}

# =========================================================
# 6. 診斷型自動修退反饋引擎 (Feedback Refinement Engine)
# =========================================================
# ---- [修正 1] 親民元素提示改為多元清單，不再獨尊「警犬」 ----------------------
_ENGAGE_HINTS = [
    "公仔", "吉祥物", "波麗士", "警察熊", "萌", "小朋友", "阿嬤", "阿公", "同仁", "員警",
    "互動", "溫馨", "青少年", "學生", "家長", "校園", "志工", "巡邏", "情境", "對話", "警犬",
]
_CTA_HINTS = ["分享", "留言", "標記", "tag", "轉發", "提醒身邊", "告訴家人", "按讚"]

def build_feedback(draft: dict, score: float, target: float, expl: dict) -> str:
    gap = (target - score) * 100
    text = draft.get("text", "")

    if score >= target:
        lines = [
            f"上一版貼文經 BERT 爆款分類器評估為 {score * 100:.1f}%，已達門檻 {target * 100:.1f}%。"
            f"本輪為「進階優化輪」，請在保留既有優點的前提下，挑戰更高的互動預測值。"
        ]
    else:
        lines = [
            f"上一版貼文經 BERT 爆款分類器評估為 {score * 100:.1f}%，距離目標門檻 {target * 100:.1f}% 尚差 {gap:.1f}%，審核未通過。"
        ]

    pos = [it["word"] for it in expl.get("positives", [])[:5]]
    neg = [it["word"] for it in expl.get("negatives", [])[:5]]
    if pos:
        lines.append(f"【保留加分項】：模型分析顯示以下詞彙具正向效益，請予以保留並加強發揮：{'、'.join(pos)}。")
    if neg:
        lines.append(f"【剔除扣分項】：以下詞彙顯著拖累了互動預測值，請刪除或替換為更生活化之語句：{'、'.join(neg)}。")

    diag = []
    if len(text) < 80:
        diag.append("內文篇幅過短，請擴充至 150～300 字，增添情境敘事或生活化小故事。")
    if len(text) > 450:
        diag.append("內文篇幅過長，請精簡至 300 字以內，將最吸睛的重點置於第一行。")
    if not re.search(r"[？?！!]", text[:60]):
        diag.append("文案開頭缺乏強烈鉤子，請以疑問句、驚嘆情境或生動對白開場。")
    if not any(h in text for h in _CTA_HINTS):
        diag.append("文末缺少行動呼籲（CTA），請加入「分享給好友」「留言提醒家人」等互動誘因。")
    if not re.search(r"#\S+", text):
        diag.append("請補上 2～4 個熱門且相關之標籤（Hashtags）。")
    if not any(h in draft.get("image_desc", "") for h in _ENGAGE_HINTS):
        diag.append("圖片描述缺乏親民元素，請依指定視覺主題融入吉祥物、員警與民眾互動、青少年或家庭等高互動視覺特徵。")

    if diag:
        lines.append("【結構與視覺優化清單】：")
        lines.extend(f"{i}. {d}" for i, d in enumerate(diag, 1))

    lines.append("請依據上述診斷，徹底重構 JSON 物件內容進行大幅度升級！")
    return "\n".join(lines)

# =========================================================
# 7. GPT-4o 生成器 (嚴格繁中提示詞工程 + 視覺主題輪替 + 臺灣制服鎖定)
# =========================================================
ZH_TEXT_RULE = (
    "畫面中若出現任何文字、看板、標語或字幕，一律使用【繁體中文（台灣正體）】，"
    "字體清晰整齊，嚴格禁止出現任何英文字母、英文單字、簡體字或拼音符號。"
)

# ---- [修正 2] 臺灣警察制服鎖定規範（依 2021 年起現行「新式警察制服」實物照片撰寫） ----
TW_POLICE_RULE = (
    "畫面中所有警察角色一律為【中華民國（臺灣）現行新式警察制服】，男女款式相同，特徵必須完全符合以下描述："
    "① 上衣：藏青深藍色（navy）機能布料立領拉鍊夾克式制服，拉鍊拉到頂、立領直挺，袖口收緊，"
    "整件為單一深藍色，沒有淺藍襯衫、沒有領帶、沒有白色或淺色元素；"
    "② 臂章：左上臂與右上臂各有一枚盾形臂章，臂章為紅色底、金黃色邊框，中央為金色警徽圖案，上緣有小字繁體中文警察局名稱；"
    "③ 胸前：左胸口有一列金色小型階級徽章（金色橫條/線條），右胸掛有黑色密錄器（隨身攝影機），"
    "肩膀處扣著黑色無線電手持麥克風並有黑色螺旋線垂下；"
    "④ 帽子：深藍色棒球帽型勤務帽，帽簷邊緣有一圈銀灰色反光滾邊，帽子正面正中央為金黃色臺灣警徽"
    "（展翅警鴿環繞國徽的圓形徽章），沒有大盤帽、沒有紅色帽牆、沒有金色帽帶；"
    "⑤ 下身：與上衣同色的藏青深藍色多口袋工作長褲（大腿兩側有貼袋），黑色戰術靴或黑色短筒皮靴；"
    "⑥ 腰部：黑色戰術勤務腰帶，配有黑色槍套、彈匣袋、手銬袋、無線電；"
    "⑦ 背景應帶有臺灣在地元素（繁體中文『○○派出所』或『○○分局』招牌、臺灣街景、機車、便利商店、學校）。"
    "嚴格禁止：中國大陸公安制服（淺藍色襯衫、深藍反光背心、『公安』字樣、紅底金星或麥穗徽章、英文 POLICE 字樣）、"
    "美國/日本/韓國/香港或任何其他國家的警察制服、白色襯衫、領帶、大盤帽、肩章上的星星。"
)

# ---- [新增] 寫實照片風格規範：以真實攝影為主，不要插畫/卡通/3D ----------------
PHOTO_STYLE_RULE = (
    "整體風格必須為【真實攝影照片】：如同專業攝影師以全片幅數位單眼相機拍攝的新聞紀實／宣傳形象照，"
    "真人、真實皮膚質感、真實布料紋理、自然光或攝影棚燈光、35mm 或 50mm 鏡頭、淺景深、高解析度、色彩自然不過飽和。"
    "嚴格禁止：插畫、卡通、動漫、Q版、3D 動畫、皮克斯風格、水彩、油畫、向量圖、CG 渲染感、塑膠質感皮膚、過度平滑的 AI 風格。"
)

# ---- [新增] 制服參考照片：若提供實物照片，圖像模型（Gemini / gpt-image-2）會以此為視覺參考，制服正確率大幅提升 ----
# 請將附件的臺灣警察制服照片放到此路徑（Colab 可上傳至 /content/），不存在時自動退回純文字模式。
UNIFORM_REF_IMAGE = "tw_police_uniform_ref.jpg"

# ---- [修正 1] 視覺主題輪替池：每個候選方案指派不同主題 ------------------------
# 全部改為「真實攝影」取向的主題，不再有插畫 / 3D / Q版
VISUAL_THEMES = [
    "真人員警走進高中校園，在教室或操場與青少年面對面互動的紀實照片",
    "紀實情境照：青少年在夜市或網咖被陌生人遞上不明包裝，員警在旁關心提醒的關鍵一幕",
    "員警在派出所門口與家長、孩子親切談心的溫馨紀實照（背景有繁體中文派出所招牌）",
    "夜間巡邏中的員警與臺灣街景、機車、便利商店霓虹燈（電影感寫實攝影）",
    "警察局宣導攤位活動現場，民眾與學生排隊拿宣導品、與員警合照的真實活動照",
    "員警與青少年一起打籃球、跑步或練拳擊，以運動取代毒品的活力紀實照",
    "員警手持繁體中文宣導看板，站在校門口或捷運站出口的形象照",
    "真人穿著警察熊/波麗士布偶裝吉祥物，與員警、小朋友在活動現場合照的真實照片",
    "青少年主角視角：學生與員警並肩站在校園走廊，員警伸手拍肩鼓勵的紀實照",
    "警犬 / 緝毒犬與領犬員警出任務的真實攝影照",   # 警犬保留為主題之一，但不再是每次都出現
]

SYSTEM_PROMPT = f"""
你是一位精通台灣警察局 Facebook 粉絲專頁經營的資深社群行銷總監與視覺導演。
你的任務是根據主題撰寫高按讚、高分享之爆款警政貼文，並提供全繁體中文之高解析度圖像與影音生成提示詞。

寫作規範：
1. 語氣親切溫馨、具共鳴感或幽默感，避免官僚宣導公文口吻。
2. 開頭第一行必須具備吸引力鉤子（情境對白、提問），結尾具備明確之行動呼籲（CTA）與 Hashtags。
3. 視覺描述具體生動，並【嚴格依照使用者分別指定之「圖片主題」與「影片主題」】設計畫面。除非指定主題本身即為警犬，
   否則畫面中不得出現警犬或緝毒犬；請善用吉祥物、員警與民眾互動、青少年、家庭、校園等多元高互動元素。

圖片與影片必須是兩個不同的故事（極其重要）：
3-1. 圖片（image_desc / image_prompt_zh）與影片（video_desc / video_prompt_zh）必須採用【完全不同的場景、人物組合、時間與情節】：
     地點不同（例如圖片在校園、影片就在夜市或派出所）、人物不同（例如圖片是員警與學生、影片就是員警與家長或青少年獨白）、
     時段與光線不同（白天 / 傍晚 / 夜晚），連標語文字也要不同。兩者只共享「宣導主題」，其餘不得重複。
3-2. 圖片是「一個定格瞬間」：一個畫面說完一件事，構圖與表情要有張力。
3-3. 影片是「一段有起承轉合的微故事」：{VIDEO_DURATION_SEC} 秒內須有【開場情境 → 轉折事件 → 員警介入或關鍵決定 → 收尾畫面/標語】四個節拍，
     請以「鏡頭 1：…／鏡頭 2：…／鏡頭 3：…／鏡頭 4：…」分鏡格式撰寫 video_prompt_zh，每個鏡頭一句，含運鏡（推進、跟拍、特寫、拉遠）。
3-4. 輸出前自我檢查：若 image_prompt_zh 與 video_prompt_zh 出現相同地點、相同人物設定或相同標語，必須改寫影片直到兩者明顯不同。

臺灣警察制服規範（極其重要）：
4. {TW_POLICE_RULE}
   image_prompt_zh 與 video_prompt_zh 中只要出現警察，就必須逐項寫出上述制服細節（深藍立領拉鍊夾克、紅底金邊盾形臂章、
   深藍棒球帽與反光滾邊、金色警徽、深藍工作褲、黑靴、密錄器、勤務腰帶），並明確標註「臺灣現行新式警察制服」。

寫實攝影規範（極其重要）：
5. {PHOTO_STYLE_RULE}
   image_prompt_zh 開頭必須以「一張真實的攝影照片，」起始；video_prompt_zh 開頭必須以「一段真實拍攝的紀實短片，」起始。

提示詞語言規範（極其重要）：
6. image_prompt_zh 與 video_prompt_zh 必須【全篇使用繁體中文撰寫】，嚴禁夾雜英文單字或描述。
7. {ZH_TEXT_RULE}
8. 提示詞需詳述：主體、動作、構圖、光影、色調、鏡頭焦段與視角、景深，並指明繁體中文標語內容。

輸出格式：必須嚴格輸出以下 6 欄位之標準 JSON：
   - "text": 貼文內文 (繁體中文)
   - "image_desc": 中文圖片特徵描述 (供 BERT 模型特徵輸入)
   - "video_desc": 中文影片動態描述 (供 BERT 模型特徵輸入)
   - "image_prompt_zh": 適合 {IMAGE_MODEL_LABEL} 的繁體中文繪圖提示詞
   - "video_prompt_zh": 適合 {VIDEO_MODEL_LABEL} 的繁體中文短影音動態提示詞
   - "video_prompt_en": video_prompt_zh 的英文分鏡摘要（3～4 句，供影片模型內部對齊用，不會顯示給使用者；
                        只描述場景/人物/動作/運鏡，不要描述任何畫面上的文字內容）
"""

def generate_post_draft_gpt4o(
    topic: str,
    publish_time: str = "星期五 20點",
    feedback: str = None,
    temperature: float = 0.7,
    visual_theme: str = None,
    video_theme: str = None,
) -> dict:
    rag_examples = retrieve_top_k_examples(topic, k=2)

    user_prompt = (
        f"【宣導主題】：{topic}\n"
        f"【預計發布時間】：{publish_time}\n"
    )
    if visual_theme:
        user_prompt += (
            f"【圖片指定主題（定格瞬間）】：{visual_theme}\n"
            f"   → image_desc / image_prompt_zh 必須圍繞此主題設計。\n"
        )
    if video_theme:
        user_prompt += (
            f"【影片指定主題（微故事，與圖片完全不同的場景與人物）】：{video_theme}\n"
            f"   → video_desc / video_prompt_zh 必須圍繞此主題設計，並以 4 個鏡頭的分鏡格式撰寫，"
            f"地點、人物、時段、標語皆不得與圖片相同。\n"
        )
    user_prompt += f"\n【歷史高互動爆款參考範例 (RAG)】：\n{rag_examples}\n"
    if feedback:
        user_prompt += f"\n⚠️【前一輪審核診斷與修改意見】：\n{feedback}\n"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=1200,
        )
        result = json.loads(response.choices[0].message.content)
        result["publish_time"] = publish_time
        result["visual_theme"] = visual_theme or ""
        result["video_theme"] = video_theme or ""
        result.setdefault("image_prompt_zh", result.get("image_prompt_en", ""))
        result.setdefault("video_prompt_zh", result.get("video_prompt_en", ""))
        result.setdefault("video_prompt_en", "")
        return result

    except Exception as e:
        print(f"❌ GPT-4o 生成失敗: {e}")
        return {
            "text": f"【{topic}宣導】平安是回家唯一的路！請大家提高警覺，注意安全。遇有任何可疑情況，請撥打 110 報案專線或 165 反詐騙專線，波麗士隨時守護您！",
            "image_desc": "親切的臺灣員警手持繁體中文防詐標語，站在派出所門口的真實照片。",
            "video_desc": "夜市裡一位阿嬤接到可疑電話，巡邏員警上前關心並協助掛斷的紀實微故事短片。",
            "image_prompt_zh": "一張真實的攝影照片，一位親切微笑的臺灣員警身穿現行新式警察制服（藏青深藍立領拉鍊夾克、紅底金邊盾形臂章、深藍棒球帽與金色警徽），雙手舉著寫有繁體中文『小心詐騙』的宣導牌，背景為有繁體中文招牌的臺灣派出所，自然光、50mm 鏡頭、淺景深。所有文字均為正體繁體中文，嚴禁出現英文。",
            "video_prompt_zh": "一段真實拍攝的紀實短片，4秒直式。鏡頭 1：傍晚臺灣夜市，一位阿嬤盯著手機神情慌張（手持跟拍）；鏡頭 2：兩位身穿臺灣現行新式深藍警察制服的巡邏員警注意到並走近（推進）；鏡頭 3：員警蹲下溫柔指著手機示意掛斷，阿嬤鬆一口氣（特寫）；鏡頭 4：三人一起笑著比出「165」手勢，畫面浮現繁體中文『可疑來電 先撥165』（拉遠）。無英文字幕。",
            "video_prompt_en": "Evening Taiwanese night market. An elderly woman stares at her phone, worried (handheld follow). Two Taiwanese police officers in navy zip-up uniforms notice and approach (push-in). One officer kneels and gently signals her to hang up; she looks relieved (close-up). The three smile together (pull back).",
            "publish_time": publish_time,
            "visual_theme": visual_theme or "",
            "video_theme": video_theme or "",
        }

# =========================================================
# 8. 多模態圖像與短影音渲染模組 (加入臺灣制服規範)
# =========================================================
def _final_visual_prompt(prompt: str, has_ref: bool = False) -> str:
    p = prompt.strip()
    if not p.startswith("一張真實的攝影照片") and not p.startswith("一段真實拍攝"):
        p = "一張真實的攝影照片，" + p
    ref_note = (
        "\n\n【參考影像說明】附上的參考照片即為臺灣現行新式警察制服的實物範例，"
        "畫面中所有警察的制服、帽子、臂章、腰帶與靴子必須與參考照片完全一致，但人物、場景與動作請依提示詞重新創作。"
        if has_ref else ""
    )
    return (
        f"{p}\n\n【攝影風格規範】{PHOTO_STYLE_RULE}"
        f"\n\n【服裝規範】{TW_POLICE_RULE}"
        f"\n\n【文字規範】{ZH_TEXT_RULE}{ref_note}"
    )

def generate_image_gpt_image_2(prompt: str, output_path: str = None) -> str:
    """呼叫 gpt-image-2 渲染寫實圖像；若有制服參考照片則改走 images.edit 以圖控圖"""
    output_path = output_path or os.path.join(OUTPUT_DIR, f"img_{datetime.now():%Y%m%d_%H%M%S}.png")
    has_ref = bool(UNIFORM_REF_IMAGE) and os.path.exists(UNIFORM_REF_IMAGE)
    full_prompt = _final_visual_prompt(prompt, has_ref=has_ref)
    print(f"🎨 正在使用 gpt-image-2 渲染寫實繁體中文圖像...（制服參考照片：{'有' if has_ref else '無'}）")
    try:
        response = None
        if has_ref:
            try:
                with open(UNIFORM_REF_IMAGE, "rb") as ref_f:
                    response = client.images.edit(
                        model="gpt-image-2", image=[ref_f], prompt=full_prompt,
                        size="1024x1024", quality="high", n=1,
                    )
            except Exception as e_ref:
                print(f"⚠️ 參考圖模式失敗 ({e_ref})，退回純文字生成。")
                response = None
        if response is None:
            response = client.images.generate(
                model="gpt-image-2", prompt=full_prompt, size="1024x1024", quality="high", n=1
            )
        item = response.data[0]
        if getattr(item, "b64_json", None):
            img_data = base64.b64decode(item.b64_json)
        elif getattr(item, "url", None):
            img_data = requests.get(item.url, timeout=120).content
        else:
            raise ValueError("未收到有效的圖像資料")

        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"✅ 圖片生成成功: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ 圖片生成失敗: {e}")
        return None


# ---- [v2.8] Gemini 3.1 Flash Image 圖像後端：以多模態對話介面生圖，參考照片作為第二個輸入 Part ----
def generate_image_gemini(prompt: str, output_path: str = None) -> str:
    """呼叫 Gemini 圖像模型渲染寫實圖像；有制服參考照片時將其與提示詞一併送入（以圖控圖）"""
    if gclient is None:
        print("❌ Gemini Client 未初始化，無法生成圖片（請設定 GEMINI_API_KEY）")
        return None
    output_path = output_path or os.path.join(OUTPUT_DIR, f"img_{datetime.now():%Y%m%d_%H%M%S}.png")
    has_ref = bool(UNIFORM_REF_IMAGE) and os.path.exists(UNIFORM_REF_IMAGE)
    full_prompt = _final_visual_prompt(prompt, has_ref=has_ref)          # 三重約束提示詞，原樣沿用
    print(f"🎨 正在使用 {IMAGE_MODEL} 渲染寫實繁體中文圖像...（制服參考照片：{'有' if has_ref else '無'}）")
    cfg = gtypes.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=gtypes.ImageConfig(aspect_ratio="1:1", image_size="1K"),   # ≈ 1024×1024
    )
    try:
        contents = [full_prompt]
        if has_ref:
            with open(UNIFORM_REF_IMAGE, "rb") as f:
                contents.append(gtypes.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))
        try:
            response = gclient.models.generate_content(model=IMAGE_MODEL, contents=contents, config=cfg)
        except Exception as e_ref:
            if not has_ref:
                raise
            print(f"⚠️ 參考圖模式失敗 ({e_ref})，退回純文字生成。")
            response = gclient.models.generate_content(model=IMAGE_MODEL, contents=[full_prompt], config=cfg)

        img_data = None
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                img_data = part.inline_data.data
                break
        if not img_data:
            raise ValueError("未收到有效的圖像資料（可能遭安全政策過濾）")
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"✅ 圖片生成成功: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ 圖片生成失敗: {e}")
        return None

def generate_image(prompt: str, output_path: str = None) -> str:
    """[v2.8] 統一入口：儀表板一律呼叫此函式，依 IMAGE_BACKEND 分派至對應後端"""
    backends = {"openai": generate_image_gpt_image_2, "gemini": generate_image_gemini}
    return backends[IMAGE_BACKEND](prompt, output_path)

def _prepare_video_reference(image_path: str, size=(720, 1280)) -> str:
    """將生成圖片裁切/縮放為影片尺寸，作為影片模型的首幀參考（Veo：image 參數；sora-2：input_reference）"""
    try:
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        tw, th = size
        scale = max(tw / im.width, th / im.height)
        im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1))
        left, top = (im.width - tw) // 2, (im.height - th) // 2
        im = im.crop((left, top, left + tw, top + th))
        ref_path = os.path.join(OUTPUT_DIR, f"video_ref_{datetime.now():%Y%m%d_%H%M%S}.png")
        im.save(ref_path)
        return ref_path
    except Exception as e:
        print(f"⚠️ 影片參考圖前處理失敗 ({e})")
        return None

def generate_video_sora(prompt: str, output_path: str = None, reference_image: str = None) -> str:
    """呼叫 sora-2 渲染寫實短影音；若提供 reference_image（通常為本次生成的圖片），
    以其作為首幀參考，讓影片中的制服與圖片一致"""
    if datetime.now() >= datetime(2026, 9, 24):
        print("❌ OpenAI 已於 2026-09-24 自 API 移除 sora-2，請將 VIDEO_BACKEND 改為 \"veo\"。")
        return None
    output_path = output_path or os.path.join(OUTPUT_DIR, f"video_{datetime.now():%Y%m%d_%H%M%S}.mp4")
    p = prompt.strip()
    if not p.startswith("一段真實拍攝"):
        p = "一段真實拍攝的紀實短片，" + p
    has_ref = bool(reference_image) and os.path.exists(reference_image)
    full_prompt = _final_visual_prompt(p, has_ref=has_ref).replace("一張真實的攝影照片，一段", "一段")
    print(f"🎬 正在使用 sora-2 渲染寫實繁體中文短影音...（首幀參考圖：{'有' if has_ref else '無'}）")
    try:
        video_job = None
        if has_ref:
            ref_path = _prepare_video_reference(reference_image)
            if ref_path:
                try:
                    with open(ref_path, "rb") as ref_f:
                        video_job = client.videos.create(
                            model="sora-2", prompt=full_prompt, size="720x1280", seconds="4", input_reference=ref_f
                        )
                except Exception as e_ref:
                    print(f"⚠️ 影片參考圖模式失敗 ({e_ref})，退回純文字生成。")
                    video_job = None
        if video_job is None:
            video_job = client.videos.create(model="sora-2", prompt=full_prompt, size="720x1280", seconds="4")
        job_id, status = video_job.id, video_job.status

        while status in ["queued", "in_progress"]:
            print("⏳ 影片渲染中，請稍候 10 秒...")
            time.sleep(10)
            video_job = client.videos.retrieve(job_id)
            status = video_job.status

        if status != "completed":
            print(f"❌ 影片渲染未成功，狀態: {status}")
            return None

        video_bytes = None
        try:
            if hasattr(client.videos, "content"):
                res = client.videos.content(job_id)
                video_bytes = res.content if hasattr(res, "content") else res.read()
        except Exception:
            pass

        if not video_bytes:
            api_key = client.api_key or os.environ.get("OPENAI_API_KEY")
            r = requests.get(
                f"https://api.openai.com/v1/videos/{job_id}/content",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=300,
            )
            if r.status_code == 200:
                video_bytes = r.content

        if video_bytes:
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            print(f"🎉 影片已儲存至: {output_path}")
            return output_path
        return None
    except Exception as e:
        print(f"❌ Sora 影片生成失敗: {e}")
        return None



# ---- [v2.8.1] Veo 專用提示詞組裝：分鏡優先、規範精簡、禁止事項改走 negative_prompt ----
# 原 _final_visual_prompt() 把 TW_POLICE_RULE（含大段「嚴格禁止…」）、攝影規範與參考影像說明整段附在分鏡之後，
# 影片模型對這種「正文短、規範長、夾雜大量否定句」的提示詞遵循度很差，容易只抓到「深藍制服的員警」就開拍，
# 劇情、地點、警犬等主體全部流失。這裡改為：分鏡放最前、制服與風格壓成兩句正向描述、禁止事項另走 negative_prompt。
VEO_UNIFORM_SHORT = (
    "所有警察皆穿著臺灣現行新式警察制服：藏青深藍色立領拉鍊夾克、雙臂紅底金邊盾形臂章、"
    "深藍色棒球帽正面金色警徽、胸前黑色密錄器、黑色勤務腰帶、深藍工作長褲與黑靴。"
)
VEO_STYLE_SHORT = "真實紀實攝影風格，真人、自然光或現場光、35mm 鏡頭、淺景深、色彩自然；畫面中若有文字一律為繁體中文。"
VEO_NEGATIVE_PROMPT = (
    "插畫, 卡通, 動漫, 3D動畫, CG渲染, 塑膠皮膚, 中國公安制服, 淺藍色襯衫, 深藍反光背心, 白色襯衫, 領帶, 大盤帽, "
    "肩章星星, 英文字幕, 英文POLICE字樣, 簡體字, 拼音, 畫面模糊, 人物變形, 多餘手指"
)

def _build_veo_prompt(video_prompt_zh: str, video_prompt_en: str = "", image_first_frame: bool = False,
                      image_prompt_zh: str = "") -> str:
    """組裝送給 Veo 的最終提示詞。
    image_first_frame=True 表示以生成圖片作為首幀（Image-to-Video）：此時影片「必定」延續圖片畫面，
    因此不再使用另一個故事的分鏡，改以圖片場景為基礎，只指示鏡頭如何動起來。"""
    if image_first_frame:
        scene = image_prompt_zh.strip() or video_prompt_zh.strip()
        body = (
            f"以這張照片為第一幀，延續同一場景、同一批人物與制服，讓畫面自然動起來："
            f"{scene}\n鏡頭緩慢推進，人物有自然的互動、對話與手勢，背景行人與車流微動。"
        )
    else:
        body = video_prompt_zh.strip()
        if not body.startswith("一段真實拍攝"):
            body = "一段真實拍攝的紀實短片，" + body
    parts = [body, f"【人物服裝】{VEO_UNIFORM_SHORT}", f"【影像風格】{VEO_STYLE_SHORT}"]
    if VEO_ADD_EN_HINT and video_prompt_en and not image_first_frame:
        parts.append(f"(Scene summary in English for the model; do not render any English text on screen): {video_prompt_en.strip()}")
    return "\n\n".join(parts)

# ---- [v2.8] Veo 3.1 影片後端：送單 → 輪詢 → 分層下載，骨架與 sora 版相同 ----
def generate_video_veo(prompt: str, output_path: str = None, reference_image: str = None,
                       prompt_en: str = "", image_prompt_zh: str = "") -> str:
    """呼叫 Veo 3.1 渲染寫實短影音；若提供 reference_image，以其作為首幀（Image-to-Video，影片將延續該畫面）"""
    if gclient is None:
        print("❌ Gemini Client 未初始化，無法生成影片（請設定 GEMINI_API_KEY）")
        return None
    output_path = output_path or os.path.join(OUTPUT_DIR, f"video_{datetime.now():%Y%m%d_%H%M%S}.mp4")
    has_ref = bool(reference_image) and os.path.exists(reference_image)
    full_prompt = _build_veo_prompt(prompt, video_prompt_en=prompt_en, image_first_frame=has_ref,
                                    image_prompt_zh=image_prompt_zh)
    print(f"🎬 正在使用 {VEO_MODEL} 渲染寫實繁體中文短影音...（首幀參考圖：{'有' if has_ref else '無'}，{VIDEO_DURATION_SEC} 秒）")
    print(f"📝 送交 Veo 之提示詞（前 300 字）：{full_prompt[:300]}")
    cfg = gtypes.GenerateVideosConfig(
        aspect_ratio="9:16", resolution="720p", duration_seconds=VIDEO_DURATION_SEC,
        number_of_videos=1, negative_prompt=VEO_NEGATIVE_PROMPT,
    )
    try:
        operation = None
        if has_ref:                                                  # Image-to-Video
            ref_path = _prepare_video_reference(reference_image)
            if ref_path:
                try:
                    with open(ref_path, "rb") as f:
                        first_frame = gtypes.Image(image_bytes=f.read(), mime_type="image/png")
                    operation = gclient.models.generate_videos(
                        model=VEO_MODEL, prompt=full_prompt, image=first_frame, config=cfg)
                except Exception as e_ref:
                    print(f"⚠️ 影片參考圖模式失敗 ({e_ref})，退回純文字生成。")
        if operation is None:                                        # Text-to-Video
            operation = gclient.models.generate_videos(model=VEO_MODEL, prompt=full_prompt, config=cfg)

        def _wait(op):
            while not op.done:
                print("⏳ 影片渲染中，請稍候 10 秒...")
                time.sleep(10)
                op = gclient.operations.get(op)
            return getattr(op.response, "generated_videos", None) or []

        videos = _wait(operation)
        if not videos and has_ref:                                   # 首幀模式遭人物政策過濾 → 改回原分鏡純文字重試一次
            print("⚠️ 首幀模式未回傳影片（可能遭人物生成政策過濾），改以純文字模式重試。")
            full_prompt = _build_veo_prompt(prompt, video_prompt_en=prompt_en, image_first_frame=False)
            videos = _wait(gclient.models.generate_videos(model=VEO_MODEL, prompt=full_prompt, config=cfg))
        if not videos:
            print("❌ 影片渲染未成功（可能遭安全政策過濾）")
            return None
        video = videos[0].video

        try:                                                         # 第一層：SDK 直接下載
            gclient.files.download(file=video, destination=output_path)
        except Exception:
            video_bytes = getattr(video, "video_bytes", None)        # 第二層：內嵌位元組
            if not video_bytes and getattr(video, "uri", None):      # 第三層：裸 HTTP
                r = requests.get(video.uri, timeout=300, headers={"x-goog-api-key": _GOOGLE_KEY})
                video_bytes = r.content if r.status_code == 200 else None
            if not video_bytes:
                return None
            with open(output_path, "wb") as f:
                f.write(video_bytes)
        print(f"🎉 影片已儲存至: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Veo 影片生成失敗: {e}")
        return None

def generate_video(prompt: str, output_path: str = None, reference_image: str = None, **extra) -> str:
    """[v2.8] 統一入口：儀表板一律呼叫此函式，依 VIDEO_BACKEND 分派至對應後端
    extra 供 Veo 後端使用（prompt_en / image_prompt_zh），sora 後端會忽略"""
    if VIDEO_BACKEND == "veo":
        return generate_video_veo(prompt, output_path, reference_image, **extra)
    return generate_video_sora(prompt, output_path, reference_image)

# =========================================================
# 9. 批次拒絕採樣與 Agent 疊代優化主排程
# =========================================================
def run_rejection_sampling(
    topic: str,
    publish_time: str = "星期五 20點",
    target_prob: float = 0.70,
    max_attempts: int = 3,
    samples_per_round: int = 3,
    min_attempts: int = 1,        # [修正 4] 最少疊代輪數
    force_full: bool = False,     # [修正 4] 強制跑完所有輪數並取全域最優
    log_callback=None,
):
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    best_draft, best_score, best_expl = None, -1.0, None
    feedback, history = None, []

    # [修正 1] 每次執行隨機打亂主題池，讓不同執行之間也有差異
    theme_pool = random.sample(VISUAL_THEMES, len(VISUAL_THEMES))

    for attempt in range(1, max_attempts + 1):
        log(f"\n🔄 [第 {attempt}/{max_attempts} 輪] — 平行生成 {samples_per_round} 個候選方案進行批次拒絕採樣")

        # 為本輪每個候選指派「圖片主題」與「影片主題」各一，且兩者必不相同；
        # 圖片主題依序輪替，影片主題取主題池中相隔一半的另一個主題，確保圖與片是兩個故事
        jobs = []
        n_pool = len(theme_pool)
        for i in range(samples_per_round):
            temp = min(0.65 + 0.12 * i + 0.05 * (attempt - 1), 1.10)
            base = (attempt - 1) * samples_per_round + i
            img_theme = theme_pool[base % n_pool]
            vid_theme = theme_pool[(base + n_pool // 2) % n_pool]
            if vid_theme == img_theme:
                vid_theme = theme_pool[(base + 1) % n_pool]
            jobs.append((temp, img_theme, vid_theme))

        # [修正 4] 真正的多執行緒平行呼叫 GPT-4o
        with ThreadPoolExecutor(max_workers=samples_per_round) as ex:
            drafts = list(ex.map(
                lambda j: generate_post_draft_gpt4o(
                    topic, publish_time=publish_time, feedback=feedback,
                    temperature=j[0], visual_theme=j[1], video_theme=j[2],
                ),
                jobs,
            ))

        bert_inputs = [build_bert_input(d) for d in drafts]
        scores = evaluate_batch(bert_inputs)
        for i, (s, (_, img_theme, vid_theme)) in enumerate(zip(scores, jobs), 1):
            log(f"   ├─ 候選方案 {i}：P(爆款) = {s * 100:.2f}%｜圖片：{img_theme[:16]}…｜影片：{vid_theme[:16]}…")

        round_idx = int(np.argmax(scores))
        round_best, round_score = drafts[round_idx], scores[round_idx]
        log(f"   └─ 本輪最優評分：{round_score * 100:.2f}%（候選 {round_idx + 1}）")

        log("🔍 正在進行反事實遮蔽法 (Occlusion XAI) 關鍵字歸因計算...")
        expl = explain_keywords(round_best)
        history.append({"attempt": attempt, "scores": scores, "best": round_score, "theme": jobs[round_idx][1], "video_theme": jobs[round_idx][2]})

        if round_score > best_score:
            best_draft, best_score, best_expl = round_best, round_score, expl

        reached = round_score >= target_prob
        if reached and attempt >= min_attempts and not force_full:
            log(f"✅ 門檻達標（{round_score * 100:.2f}% ≥ {target_prob * 100:.1f}%），且已滿足最少輪數 {min_attempts}，通過審核，終止疊代！")
            break

        if attempt == max_attempts:
            break

        if reached:
            log(f"✅ 已達門檻（{round_score * 100:.2f}%），但依設定繼續進階優化（最少 {min_attempts} 輪 / 強制完整={force_full}）...")
        else:
            log("⚠️ 未達門檻，啟動自動修退機制 (Refinement)，組裝診斷意見反饋給 GPT-4o...")
        feedback = build_feedback(round_best, round_score, target_prob, expl)

    if best_expl is None and best_draft is not None:
        best_expl = explain_keywords(best_draft)

    # [修正 4] 各輪成績總表
    log("\n📈 各輪成績總表：")
    for h in history:
        marks = " / ".join(f"{s * 100:.1f}%" for s in h["scores"])
        log(f"   第 {h['attempt']} 輪：候選 [{marks}] → 最優 {h['best'] * 100:.2f}%")
    log(f"   全域最優：{best_score * 100:.2f}%")

    return best_draft, best_score, best_expl, history

# =========================================================
# 10. Gradio 跨版本相容層與介面元件
# =========================================================
def _init_params(cls_or_fn) -> set:
    try:
        target = cls_or_fn.__init__ if isinstance(cls_or_fn, type) else cls_or_fn
        return set(inspect.signature(target).parameters.keys())
    except (TypeError, ValueError):
        return set()

def _make(cls, *args, **kwargs):
    while True:
        try:
            return cls(*args, **kwargs)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
            if m and m.group(1) in kwargs:
                kwargs.pop(m.group(1))
                continue
            raise

_TEXTBOX_PARAMS = _init_params(gr.Textbox)
_LAUNCH_PARAMS = _init_params(gr.Blocks.launch)
_BLOCKS_PARAMS = _init_params(gr.Blocks)
DownloadButtonCls = getattr(gr, "DownloadButton", gr.File)

def copy_textbox(**kwargs):
    if "buttons" in _TEXTBOX_PARAMS:
        kwargs["buttons"] = ["copy"]
    else:
        kwargs["show_copy_button"] = True
    return _make(gr.Textbox, **kwargs)

def safe_launch(demo, **kwargs):
    while True:
        try:
            return demo.launch(**kwargs)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
            if m and m.group(1) in kwargs:
                kwargs.pop(m.group(1))
                continue
            raise

def make_gauge(prob: float, target: float):
    pct = prob * 100
    bar_color = "#1E8449" if prob >= target else ("#C8A24A" if prob >= target * 0.7 else "#C0392B")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "color": "#0F2942"}},
            title={"text": "BERT 爆款預測機率", "font": {"size": 16, "color": "#0F2942"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#0F2942"},
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "#FADBD8"},
                    {"range": [40, 70], "color": "#FCF3CF"},
                    {"range": [70, 100], "color": "#D5F5E3"},
                ],
                "threshold": {"line": {"color": "#0F2942", "width": 4}, "thickness": 0.85, "value": target * 100},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def render_xai_html(expl: dict) -> str:
    if not expl or not expl.get("items"):
        return "<p style='color:#888'>尚無分析結果。</p>"

    def chips(items, color, sign):
        if not items:
            return "<span style='color:#888'>（無明顯特徵）</span>"
        out = []
        for it in items:
            val = abs(it["contribution"]) * 100
            out.append(
                f"<span style='display:inline-block;margin:3px;padding:4px 9px;"
                f"border-radius:12px;background:{color}1A;border:1px solid {color};"
                f"color:{color};font-size:12px;'>"
                f"{it['word']} <b>{sign}{val:.2f}%</b>"
                f"<span style='opacity:.65'>·{it['section']}</span></span>"
            )
        return "".join(out)

    return f"""
    <div style='font-family:Noto Sans TC,sans-serif;line-height:1.8'>
      <p style='margin:0 0 6px;font-size:13px;'>基準預測機率 <b>{expl['base_prob']*100:.2f}%</b>。數值＝移除該詞彙後機率的變化量（正值代表該詞有助於爆款）：</p>
      <div style='margin-bottom:6px'><b style='color:#1E8449'>🟢 爆款正向貢獻詞（建議強化保留）：</b><br>{chips(expl.get('positives', []), '#1E8449', '+')}</div>
      <div><b style='color:#C0392B'>🔴 負向拖累詞彙（建議刪除改寫）：</b><br>{chips(expl.get('negatives', []), '#C0392B', '-')}</div>
    </div>
    """

CUSTOM_CSS = """
.gradio-container {font-family: 'Noto Sans TC','Microsoft JhengHei',sans-serif !important;}
#hero {
    background: linear-gradient(135deg, #F0F4F8 0%, #E6EEF8 100%) !important;
    padding: 22px 26px !important; border-radius: 12px !important; margin-bottom: 14px !important;
    border-left: 6px solid #1B3A5C !important; border-top: 1px solid #CBD5E1 !important;
    border-right: 1px solid #CBD5E1 !important; border-bottom: 1px solid #CBD5E1 !important;
}
#hero h1 {margin: 0 !important; color: #0F2942 !important; font-size: 23px !important; font-weight: 700 !important; letter-spacing: 0.5px !important;}
#hero p {margin: 8px 0 0 0 !important; color: #334155 !important; font-size: 14px !important; font-weight: 500 !important; line-height: 1.5 !important;}
"""

def _empty_outputs(msg, target_prob, xai_msg):
    return (msg, make_gauge(0, target_prob), "", "", "", "", [], xai_msg, None, gr.update(visible=False), None, gr.update(visible=False))

def ui_generate_stream(
    topic, publish_time, target_prob, max_attempts, samples_per_round, min_attempts, force_full, do_image, do_video,
    video_use_image_ref=False,
):
    logs = []
    def push(msg):
        logs.append(msg)

    if not topic or not topic.strip():
        yield _empty_outputs("⚠️ 請先輸入宣導主題！", target_prob, "<p>無資料</p>")
        return

    push(f"🚀 啟動主題生成任務：『{topic}』")
    yield _empty_outputs("\n".join(logs), target_prob, "<p>分析中...</p>")

    best_draft, best_score, expl, _ = run_rejection_sampling(
        topic=topic.strip(),
        publish_time=publish_time,
        target_prob=float(target_prob),
        max_attempts=int(max_attempts),
        samples_per_round=int(samples_per_round),
        min_attempts=int(min_attempts),
        force_full=bool(force_full),
        log_callback=push,
    )

    verdict = "✅ 通過門檻" if best_score >= target_prob else "⚠️ 未達門檻（建議人工微調）"
    push(f"\n🏆 最優方案出爐 [{verdict}]｜P(Good) = {best_score * 100:.2f}%")
    push(f"   🖼️ 圖片主題：{best_draft.get('visual_theme', '')}")
    push(f"   🎬 影片主題：{best_draft.get('video_theme', '')}")

    bert_feat_str = (
        f"【圖片主題】{best_draft.get('visual_theme', '')}\n"
        f"【影片主題】{best_draft.get('video_theme', '')}\n\n"
        f"【圖片特徵】{best_draft.get('image_desc', '')}\n\n"
        f"【影片特徵】{best_draft.get('video_desc', '')}"
    )

    def pack(img_path=None, vid_path=None):
        return (
            "\n".join(logs),
            make_gauge(best_score, target_prob),
            best_draft.get("text", ""),
            best_draft.get("image_prompt_zh", ""),
            best_draft.get("video_prompt_zh", ""),
            bert_feat_str,
            expl.get("highlight", []),
            render_xai_html(expl),
            img_path, gr.update(value=img_path, visible=bool(img_path)),
            vid_path, gr.update(value=vid_path, visible=bool(vid_path)),
        )

    yield pack()

    img_path, vid_path = None, None
    if do_image and best_draft.get("image_prompt_zh"):
        push(f"\n🎨 呼叫 {IMAGE_MODEL_LABEL} 進行寫實圖像渲染中（已注入臺灣新式警察制服規範與寫實攝影規範）...")
        yield pack()
        img_path = generate_image(best_draft["image_prompt_zh"])
        push("✅ 圖像渲染完成！" if img_path else "❌ 圖像生成失敗。")
        yield pack(img_path)

    if do_video and best_draft.get("video_prompt_zh"):
        # 影片與圖片是不同故事，因此預設「不」以圖片作為首幀參考；
        # 只有勾選 video_use_image_ref 時才沿用圖片場景（適合想要圖/片同一畫面延伸的情況）
        use_ref = bool(video_use_image_ref) and bool(img_path)
        ref_msg = ("，並以本次生成圖片作為首幀（影片將延續圖片場景，不採用另一個影片故事）"
                   if use_ref else "，採用與圖片不同的故事與場景（影片提示詞分鏡）")
        push(f"\n🎬 呼叫 {VIDEO_MODEL_LABEL} 進行寫實短影音渲染中（約需 1～3 分鐘，{VIDEO_DURATION_SEC} 秒，已注入臺灣警察制服規範{ref_msg}）...")
        yield pack(img_path)
        vid_path = generate_video(
            best_draft["video_prompt_zh"], reference_image=img_path if use_ref else None,
            prompt_en=best_draft.get("video_prompt_en", ""), image_prompt_zh=best_draft.get("image_prompt_zh", ""),
        )
        push("✅ 短影音渲染完成！" if vid_path else "❌ 影片生成失敗。")

    yield pack(img_path, vid_path)

# =========================================================
# 11. 建立 UI 佈局
# =========================================================
def build_ui():
    style = {"css": CUSTOM_CSS, "title": "智慧警政 FB 爆款生成雙引擎系統", "theme": gr.themes.Soft()}
    if "css" in _LAUNCH_PARAMS:
        blocks_kwargs, launch_kwargs = {}, style
    else:
        blocks_kwargs, launch_kwargs = style, {}
    blocks_kwargs = {k: v for k, v in blocks_kwargs.items() if k in _BLOCKS_PARAMS or not _BLOCKS_PARAMS}

    with _make(gr.Blocks, **blocks_kwargs) as demo:
        gr.HTML(
            """
            <div id='hero' style='background: linear-gradient(135deg, #F0F4F8 0%, #E6EEF8 100%); padding: 22px 26px; border-radius: 12px; margin-bottom: 14px; border-left: 6px solid #1B3A5C; border-top: 1px solid #CBD5E1; border-right: 1px solid #CBD5E1; border-bottom: 1px solid #CBD5E1;'>
                <h1 style='margin: 0; color: #0F2942 !important; font-size: 23px; font-weight: 700; letter-spacing: 0.5px;'>👮‍♂️ 智慧警政 FB 爆款內容 AI 生成與評估雙引擎系統</h1>
                <p style='margin: 8px 0 0 0; color: #334155 !important; font-size: 14px; font-weight: 500; line-height: 1.5;'>整合 GPT-4o 生成式 Agent × 預適配 RAG × BERT 爆款審核 × 批次拒絕採樣 × 遮蔽法 XAI 特徵歸因 × Gemini／Veo 多模態渲染</p>
            </div>
            """
        )

        with gr.Row():
            topic = _make(gr.Textbox, label="🎯 宣導主題", placeholder="請輸入主題（如：寒假防詐、酒後不開車、反毒宣導）", scale=3)
            publish_time = _make(gr.Textbox, label="🕒 預計發布時間", value="星期五 20點", scale=1)

        with gr.Row():
            target_prob = _make(gr.Slider, minimum=0.5, maximum=0.95, value=0.70, step=0.01, label="🛡️ 爆款審核門檻 P(Good)")
            max_attempts = _make(gr.Slider, minimum=1, maximum=5, value=3, step=1, label="🔄 最多疊代優化輪數")
            min_attempts = _make(gr.Slider, minimum=1, maximum=5, value=1, step=1, label="⏱️ 最少疊代輪數（達標也至少跑滿）")
            samples_per_round = _make(gr.Slider, minimum=1, maximum=5, value=3, step=1, label="🎲 每輪候選數 (拒絕採樣)")

        with gr.Row():
            force_full = _make(gr.Checkbox, label="🏁 強制跑完所有輪數並取全域最優（不因達標提前停止）", value=False)
            do_image = _make(gr.Checkbox, label=f"🖼️ 開啟 API 自動繪圖 ({IMAGE_MODEL_LABEL})", value=False)
            do_video = _make(gr.Checkbox, label=f"🎥 開啟 API 自動產生影片 ({VIDEO_MODEL_LABEL})", value=False)
            video_use_image_ref = _make(gr.Checkbox, label="🔗 影片沿用圖片場景（以圖片為首幀；預設關閉＝圖/片為不同故事）", value=False)

        run_btn = _make(gr.Button, "🚀 啟動 AI 閉環生成與驗證", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 📊 預測評估與文案推薦")
                gauge = _make(gr.Plot, label="高互動機率量規圖")
                post_text = copy_textbox(label="📝 推薦貼文內文（支援一鍵複製與編輯）", lines=10, interactive=True)
                img_prompt = copy_textbox(label=f"🎨 繁體中文繪圖提示詞 ({IMAGE_MODEL_LABEL})", lines=4, interactive=True)
                vid_prompt = copy_textbox(label=f"🎬 繁體中文影片提示詞 ({VIDEO_MODEL_LABEL})", lines=4, interactive=True)
                bert_feat = copy_textbox(label="🔍 視覺主題與 BERT 視覺/動態特徵描述", lines=4)

            with gr.Column(scale=5):
                gr.Markdown("### 🖼️ 多模態生成成果預覽與下載")
                out_image = _make(gr.Image, label="生成圖片預覽 (1024×1024)", type="filepath", height=380)
                dl_image = _make(DownloadButtonCls, label="📥 下載高解析度圖片", visible=False)
                out_video = _make(gr.Video, label=f"生成短影音預覽 (720p 直式 9:16，{VIDEO_DURATION_SEC} 秒)", height=380)
                dl_video = _make(DownloadButtonCls, label="📥 下載短影音檔案", visible=False)

        with gr.Accordion("🧠 可解釋性 AI（XAI）遮蔽法特徵歸因面板 — 人工最終微調參考", open=True):
            # [修正 3] 分類標籤模式 + 自訂 color_map，綠＝拉高、紅＝拖累
            xai_highlight = _make(
                gr.HighlightedText,
                label="內文逐詞貢獻度 (深綠/淺綠＝拉高爆款機率 / 淺紅/深紅＝拖累)",
                color_map=XAI_COLOR_MAP,
                show_legend=True,
                combine_adjacent=False,
            )
            xai_html = _make(gr.HTML)

        with gr.Accordion("📋 疊代優化歷程日誌 (批次拒絕採樣與自動修退)", open=False):
            log_box = copy_textbox(label="執行紀錄", lines=16)

        run_btn.click(
            fn=ui_generate_stream,
            inputs=[topic, publish_time, target_prob, max_attempts, samples_per_round, min_attempts, force_full, do_image, do_video, video_use_image_ref],
            outputs=[log_box, gauge, post_text, img_prompt, vid_prompt, bert_feat, xai_highlight, xai_html, out_image, dl_image, out_video, dl_video],
        )

    return demo, launch_kwargs

# =========================================================
# 12. 系統啟動
# =========================================================
if __name__ == "__main__":
    app, launch_style = build_ui()
    safe_launch(app, share=True, debug=False, **launch_style)

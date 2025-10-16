import streamlit as st
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import datetime

# 現在の和暦年を取得（令和）
current_year = datetime.datetime.now().year
reiwa_year = current_year - 2018
reiwa_tags = [f"#令和{reiwa_year - 2}年ベビー", f"#令和{reiwa_year - 1}年ベビー", f"#令和{reiwa_year}年ベビー"]

# 形態素解析器
tokenizer = Tokenizer()

# ストップワード
stopwords = set([
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'して', 'ます', 'です',
    'いる', 'ある', 'する', 'なる', '月', '時', '日', '〜', '・', '※', 'から',
    'ごろ', 'まで', '毎日', '午前', '午後', '内容', '曜日', '予約', '受付',
    '参加', '開催', '学区', 'お話', '団体名', 'イベント', '対象年齢', '終了',
    'ほり', 'ほりオープン', 'オープン', 'オープンハート', 'ハート', 'エリア',
    'ゆり', 'かご', '部屋', 'たち'
])

fixed_priority_keywords = set([
    'ゆりかご', 'すまいる',
    '子育て', '子育て支援', '瑞穂区', '名古屋ママ', '地域子育て支援拠点',
    '子育て応援拠点', 'さくらっこ♪', '子育てネットワークさくらっこ♪',
    '未就園児親子', '0歳ママ', '1歳ママ', '2歳ママ', '3歳ママ',
    '未就園児', '子育て広場', '子育て相談', 'マタニティ', '妊婦',
    'プレママ', '新米ママ', *reiwa_tags,
    '絵本', '手遊び', '広場', 'ランチ', '親子', '一時預かり', 'イベント'
])

always_include_hashtags = list(dict.fromkeys([
    '#子育て', '#子育て支援', '#瑞穂区', '#名古屋ママ',
    '#地域子育て支援拠点', '#子育て応援拠点', '#さくらっこ♪',
    '#子育てネットワークさくらっこ♪', '#瑞穂区ママ',
    '#プレママ', '#新米ママ', '#マタニティ', '#妊婦',
    *reiwa_tags
]))

def is_valid_word(word):
    return (
        word not in stopwords and
        not re.fullmatch(r'\d+', word) and
        not re.fullmatch(r'[a-zA-Z]+', word) and
        not re.fullmatch(r'[\W_]+', word) and
        len(word) > 1
    )

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if (pos in ['名詞', '固有名詞']) and is_valid_word(base):
            words.append(base)
    return words

def extract_keywords(text, top_n=15):
    words = tokenize(text)
    if not words:
        return []

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x, lowercase=False, ngram_range=(1, 2)
    )
    tfidf = vectorizer.fit_transform([words])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

    fixed_in_text = [kw for kw in fixed_priority_keywords if kw in text]
    extracted = fixed_in_text[:]

    for word, score in sorted_words:
        if word not in extracted and len(extracted) < top_n:
            extracted.append(word)
    return extracted[:top_n]

def extract_org_hashtag(text):
    match = re.search(r'団体名[:：]\s*([^\n\r]+)', text)
    if match:
        org_name = match.group(1).strip()
        clean_name = re.sub(r'[^\wぁ-んァ-ヶ一-龥ー♡♥ー・＆A-Za-z0-9]', '', org_name)
        return '#' + clean_name
    return None

def extract_title_hashtag(text):
    match = re.search(r'「\s*\d+月\s+(.+?)」を開催しました', text)
    if match:
        raw = match.group(1).strip()
        clean = re.sub(r'[^\wぁ-んァ-ヶ一-龥ー♡♥ー・＆A-Za-z0-9]', '', raw)
        return '#' + clean
    return None

def generate_hashtags(text, selected_fixed_tags, top_n=15):
    keywords = extract_keywords(text, top_n)
    auto_tags = ['#' + re.sub(r'\s+', '', kw) for kw in keywords]

    title_tag = extract_title_hashtag(text)
    org_tag = extract_org_hashtag(text)
    for tag in [title_tag, org_tag]:
        if tag and tag not in auto_tags:
            auto_tags = [tag] + auto_tags

    fixed_tags = [tag for tag in selected_fixed_tags if tag not in auto_tags]
    return auto_tags + fixed_tags

# --- Streamlit UI ---
st.title("子育て投稿向けハッシュタグ自動生成アプリ")

user_input = st.text_area("投稿内容を入力してください：", height=200)

st.markdown("### 固定ハッシュタグを選択してください：")
num_columns = 3
cols = st.columns(num_columns)
selected_fixed_tags = []
for i, tag in enumerate(always_include_hashtags):
    col = cols[i % num_columns]
    with col:
        if st.checkbox(tag, value=True, key=f"fixed_tag_{i}"):
            selected_fixed_tags.append(tag)

# ハッシュタグ生成
if st.button("✅ ハッシュタグを生成"):
    if user_input.strip():
        tags = generate_hashtags(user_input, selected_fixed_tags)
        tags = list(dict.fromkeys(tags))  # 重複削除
        st.session_state["post_text"] = user_input.strip()
        st.session_state["hashtags_selected"] = {tag: True for tag in tags}
    else:
        st.warning("投稿文を入力してください。")

# --------------------------
# 編集エリア
# --------------------------
if "post_text" in st.session_state:
    st.markdown("### ハッシュタグ編集（チェックを外すと削除されます）")
    hashtags_selected = st.session_state.get("hashtags_selected", {})

    # すべてのタグをチェックボックス表示
    all_tags = list(hashtags_selected.keys())
    cols_edit = st.columns(3)
    for i, tag in enumerate(all_tags):
        col = cols_edit[i % 3]
        with col:
            checked = st.checkbox(tag, value=hashtags_selected.get(tag, True), key=f"edit_{tag}")
            hashtags_selected[tag] = checked

    # --- 新しいタグの追加 ---
    with st.form(key="add_tag_form"):
        new_tag_input = st.text_input("新しいハッシュタグを追加（#無しでOK）", key="new_tag_input")
        submitted = st.form_submit_button("追加")
        if submitted:
            new_tag = new_tag_input.strip()
            if new_tag:
                if not new_tag.startswith("#"):
                    new_tag = "#" + new_tag
                hashtags_selected[new_tag] = True  # 追加したら即チェックON
            st.rerun()

    st.session_state["hashtags_selected"] = hashtags_selected

    # 選択されているタグだけを反映
    final_tags = [tag for tag, selected in hashtags_selected.items() if selected]
    preview_text = st.session_state["post_text"] + "\n\n" + " ".join(final_tags)

    # --- ハッシュタグ数の表示 ---
    hashtag_count = len(final_tags)
    if hashtag_count < 20:
        st.markdown(f"<div style='background-color:#FFF3CD;color:#856404;padding:10px;border-radius:8px;'>⚠️ 現在のハッシュタグ数：<b>{hashtag_count}個</b>（少なめ）<br>20個以上つけると効果的です！</div>", unsafe_allow_html=True)
    elif hashtag_count > 25:
        st.markdown(f"<div style='background-color:#D1ECF1;color:#0C5460;padding:10px;border-radius:8px;'>ℹ️ 現在のハッシュタグ数：<b>{hashtag_count}個</b>（多め）<br>やや多いかもしれませんが効果は高い可能性があります。</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#D4EDDA;color:#155724;padding:10px;border-radius:8px;'>✅ 現在のハッシュタグ数：<b>{hashtag_count}個</b>（ちょうどいい）</div>", unsafe_allow_html=True)

    # --- コピー機能 ---
    js_code = f"""
    <script>
    function copyToClipboard(text) {{
        navigator.clipboard.writeText(text).then(function() {{
            alert("コピーしました");
        }});
    }}
    </script>
    <div style="text-align: right; margin-bottom: -10px; margin-top: -5px;">
    <button onclick="copyToClipboard(`{preview_text}`)">📋 コピー</button>
    </div>
    """
    st.markdown("### 👀 プレビュー（投稿文＋ハッシュタグ）")
    st.components.v1.html(js_code, height=40)
    st.text_area("投稿文＋ハッシュタグ：", value=preview_text, height=180, disabled=True)

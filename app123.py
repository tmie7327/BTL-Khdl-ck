# -*- coding: utf-8 -*-
"""
Ứng dụng Streamlit: Tóm tắt ý nghĩa các Cluster từ dữ liệu khảo sát (dầu nhớt).
Chạy: streamlit run app.py
Cài đặt: pip install streamlit pandas openai openpyxl
"""
from __future__ import annotations

import io
import json
import re
import random
import time
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from openai import OpenAI

# ---------------------------------------------------------------------------
# CẤU HÌNH TỔNG QUÁT
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# Giới hạn để tránh tràn token: chỉ gửi tối đa một phần câu trả lời.
MAX_CHARS_PER_ANSWER = 1_200
MAX_TOTAL_ANSWERS_CHARS = 30_000

# Retry/backoff khi gặp rate limit/quota
MAX_API_RETRIES = 6
SECONDS_BETWEEN_CLUSTER_CALLS = 0.2


def is_plausible_openai_api_key(key: str) -> bool:
    """Kiểm tra sơ bộ để tránh người dùng quên dán API key thật."""
    if not key or not key.strip():
        return False
    k = key.strip()
    # Khóa OpenAI thường có dạng sk-...
    return k.startswith("sk-") or len(k) > 25


def _truncate_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ..."


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def build_openai_prompt(cluster_id: object, sampled_answers: list[str]) -> str:
    # Cắt bớt từng câu trả lời để không làm prompt quá dài.
    trimmed = [_truncate_chars(a, MAX_CHARS_PER_ANSWER) for a in sampled_answers]

    answers_block_lines = []
    total_chars = 0
    for i, a in enumerate(trimmed, start=1):
        line = f"{i}. {a}"
        if total_chars + len(line) > MAX_TOTAL_ANSWERS_CHARS:
            remaining = max(0, MAX_TOTAL_ANSWERS_CHARS - total_chars)
            line = line[:remaining].rstrip() + " ..."
            answers_block_lines.append(line)
            break
        answers_block_lines.append(line)
        total_chars += len(line)

    answers_block = "\n".join(answers_block_lines)

    return (
        "Dựa vào các câu trả lời sau, hãy đặt một tiêu đề ngắn gọn cho nhóm này và "
        "tóm tắt 3 điểm chính mà người dùng đang quan tâm.\n"
        f"(Cluster: {cluster_id})\n\n"
        "CÁC CÂU TRẢ LỜI:\n"
        f"{answers_block}\n\n"
        "Hãy TRẢ VỀ CHỈ DUY NHẤT JSON với cấu trúc:\n"
        '{"title": "<tiêu đề ngắn>", "summary": "<tóm tắt 3 ý chính>"}\n'
        "Trong summary: mỗi ý trên một dòng bắt đầu bằng dấu '-' (tổng cộng 3 dòng). "
        "KHÔNG kèm giải thích ngoài JSON."
    )


def summarize_cluster_text(
    cluster_id: object,
    sampled_answers: list[str],
    client: OpenAI,
    model_name: str,
) -> tuple[str, str]:
    """
    Trả về (title, summary) cho một cluster bằng OpenAI.
    """
    prompt = build_openai_prompt(cluster_id, sampled_answers)
    system_msg = (
        "Bạn là chuyên gia phân tích ý kiến khảo sát. Trả lời tiếng Việt. "
        "Luôn tuân thủ format JSON và không giải thích thêm."
    )

    last_error: str | None = None
    for attempt in range(MAX_API_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            data = _extract_json_object(content)
            if not data:
                return "Không xác định", content.strip()[:5000] or "Không nhận được tóm tắt."

            title = str(data.get("title", "")).strip() or "Không xác định"
            summary = str(data.get("summary", "")).strip()
            return title, summary
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            wait_s = min(60.0, (2**attempt) + (0.25 * random.random()))
            time.sleep(wait_s)

    return "Lỗi tóm tắt", f"API lỗi sau {MAX_API_RETRIES} lần thử: {last_error}"


def load_excel_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Đọc Excel từ bytes thành DataFrame."""
    return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")


def load_excel(uploaded_file) -> pd.DataFrame:
    """Đọc file Excel từ bộ nhớ (upload Streamlit) thành DataFrame."""
    return load_excel_from_bytes(uploaded_file.getvalue())


def load_excel_from_url(url: str) -> pd.DataFrame:
    """Tải file Excel từ URL rồi parse bằng pandas."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        content = resp.read()
    return load_excel_from_bytes(content)


def validate_columns(df: pd.DataFrame, cluster_col: str, answer_col: str) -> tuple[bool, str]:
    """Kiểm tra các cột bắt buộc có tồn tại không."""
    missing = [c for c in (cluster_col, answer_col) if c not in df.columns]
    if missing:
        cols_preview = ", ".join(df.columns.astype(str).tolist())
        return False, (
            f"Thiếu cột: {', '.join(missing)}. "
            f"Các cột hiện có: {cols_preview}"
        )
    return True, ""


def display_cluster_id(val: object) -> str:
    """Hiển thị Cluster ID thân thiện cho NaN."""
    if pd.isna(val):
        return "(Cluster trống / NaN)"
    return str(val).strip()


def build_cluster_text_series(
    df: pd.DataFrame,
    cluster_col: str,
    answer_col: str,
    min_sample_n: int,
    max_sample_n: int,
    seed: int,
) -> list[dict]:
    """
    Group by Cluster ID, rồi lấy ngẫu nhiên N câu trả lời đại diện (N trong [min_sample_n, max_sample_n]).
    """
    if min_sample_n < 1 or max_sample_n < min_sample_n:
        raise ValueError("min_sample_n và max_sample_n không hợp lệ.")

    work = df[[cluster_col, answer_col]].copy()
    work[answer_col] = work[answer_col].apply(
        lambda x: str(x).strip() if pd.notna(x) else ""
    )
    work = work[work[answer_col] != ""]

    rng = random.Random(int(seed))
    clusters: list[dict] = []
    for cluster_val, g in work.groupby(cluster_col, dropna=False):
        answers = g[answer_col].tolist()
        if not answers:
            continue

        n = rng.randint(int(min_sample_n), int(max_sample_n))
        n = min(n, len(answers))
        sampled = rng.sample(answers, n)
        clusters.append({"cluster_id": cluster_val, "sampled_answers": sampled})

    return clusters


def main() -> None:
    st.set_page_config(page_title="Tóm tắt Cluster — Khảo sát", layout="wide")

    with st.sidebar:
        st.header("Cấu hình OpenAI")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Dán API key của bạn tại đây.",
        )
        model_name = st.text_input("Model", value=DEFAULT_MODEL_NAME)

        st.divider()
        st.subheader("Lấy mẫu câu trả lời")
        min_sample_n = st.number_input(
            "Số câu tối thiểu / cluster",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
        )
        max_sample_n = st.number_input(
            "Số câu tối đa / cluster",
            min_value=1,
            max_value=200,
            value=15,
            step=1,
        )
        seed = st.number_input("Seed (tái lập)", min_value=0, max_value=1_000_000_000, value=42, step=1)

        st.divider()
        max_clusters = st.number_input(
            "Giới hạn số cluster xử lý (0 = tất cả)",
            min_value=0,
            max_value=10_000,
            value=0,
            step=1,
        )

    st.title("Tóm tắt ý nghĩa Cluster từ dữ liệu khảo sát")
    st.caption("Mỗi cluster lấy ngẫu nhiên từ 10–15 câu trả lời đại diện để tránh tràn token.")

    data_source = st.radio("Nguồn dữ liệu", ["Upload Excel", "Tải từ URL"], horizontal=True)

    df: pd.DataFrame | None = None
    if data_source == "Upload Excel":
        uploaded = st.file_uploader("Chọn file Excel (.xlsx)", type=["xlsx", "xls"])
        if uploaded is not None:
            try:
                df = load_excel(uploaded)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Không đọc được file Excel: {type(exc).__name__}: {exc}")
                return
    else:
        url = st.text_input("Nhập URL file Excel (.xlsx)")
        if st.button("Tải từ URL", type="secondary", disabled=not bool(url.strip())):
            try:
                with st.spinner("Đang tải file từ URL..."):
                    df = load_excel_from_url(url.strip())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Tải/parse file thất bại: {type(exc).__name__}: {exc}")
                return

    if df is None:
        st.info("Chưa có dữ liệu. Hãy upload file hoặc nhập URL để bắt đầu.")
        return

    st.subheader("Xem trước dữ liệu (5 dòng đầu)")
    st.dataframe(df.head(5), use_container_width=True)

    cols = [str(c) for c in df.columns.tolist()]

    def pick_default_col(candidates: list[str]) -> str:
        for cand in candidates:
            for c in cols:
                if c.strip().lower() == cand.strip().lower():
                    return c
        return cols[0]

    default_cluster = pick_default_col(["Cluster", "Cluster ID", "cluster", "cluster_id"])
    default_answer = pick_default_col(["Content", "Câu trả lời khảo sát", "Answer", "Text", "response"])

    cluster_col = st.selectbox("Cột Cluster ID", options=cols, index=cols.index(default_cluster) if default_cluster in cols else 0)
    answer_col = st.selectbox("Cột câu trả lời khảo sát", options=cols, index=cols.index(default_answer) if default_answer in cols else 0)

    ok, err_msg = validate_columns(df, cluster_col=cluster_col, answer_col=answer_col)
    if not ok:
        st.error(err_msg)
        return

    analyze = st.button("Phân tích và Tóm tắt Cluster", type="primary")
    if not analyze:
        return

    if not is_plausible_openai_api_key(api_key):
        st.error("Thiếu/không hợp lệ OpenAI API Key. Hãy dán đúng key trong sidebar.")
        return
    if int(min_sample_n) > int(max_sample_n):
        st.error("Số câu tối thiểu không được lớn hơn số câu tối đa.")
        return

    clusters = build_cluster_text_series(
        df=df,
        cluster_col=cluster_col,
        answer_col=answer_col,
        min_sample_n=int(min_sample_n),
        max_sample_n=int(max_sample_n),
        seed=int(seed),
    )

    if not clusters:
        st.warning("Không có câu trả lời hợp lệ để tóm tắt (cột câu trả lời có thể đang trống).")
        return

    if int(max_clusters) > 0 and len(clusters) > int(max_clusters):
        clusters = clusters[: int(max_clusters)]

    client = OpenAI(api_key=api_key.strip())

    st.subheader("Kết quả tóm tắt theo từng Cluster")
    progress = st.progress(0)
    results_rows: list[dict] = []

    with st.spinner("Đang gọi OpenAI để tóm tắt từng cluster, vui lòng chờ..."):
        for idx, item in enumerate(clusters, start=1):
            if idx > 1:
                time.sleep(SECONDS_BETWEEN_CLUSTER_CALLS)

            cluster_id = item["cluster_id"]
            sampled_answers = item["sampled_answers"]

            title, summary = summarize_cluster_text(
                cluster_id=cluster_id,
                sampled_answers=sampled_answers,
                client=client,
                model_name=model_name.strip(),
            )

            results_rows.append(
                {
                    "Cluster ID": display_cluster_id(cluster_id),
                    "Tên chủ đề": title,
                    "Tóm tắt": summary,
                }
            )

            progress.progress(min(1.0, idx / len(clusters)))

    results_df = pd.DataFrame(results_rows, columns=["Cluster ID", "Tên chủ đề", "Tóm tắt"])
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Tải kết quả")

    csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Tải CSV",
        data=csv_bytes,
        file_name="cluster_summary.csv",
        mime="text/csv",
    )

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Results")

    st.download_button(
        label="Tải Excel",
        data=excel_buf.getvalue(),
        file_name="cluster_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()

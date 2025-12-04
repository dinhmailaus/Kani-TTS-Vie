import os
import re
import tempfile
import time
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf

from kani_vie.tts_core import Config, KaniModel, NemoAudioPlayer
from utils.normalize_text import VietnameseTTSNormalizer

# --- Speaker options ---
SPEAKER_CHOICES = [
    ("Khoa – Nam miền Bắc", "nam-mien-bac"),
    ("Hùng – Nam miền Nam", "nam-mien-nam"),
    ("Trinh – Nữ miền Nam", "nu-mien-nam"),
    ("David – English (British)", "david"),
    ("Katie – English (Irish)", "katie"),
    ("Không chỉ định", None),
]

# --- Text limits ---
MAX_TEXT_LEN = 8000          # tối đa 8000 ký tự cho toàn bộ input
MAX_CHARS_PER_CHUNK = 250    # mỗi đoạn gửi vào mô hình


# --- Initialize model once ---
def _init_models():
    config = Config()
    player = NemoAudioPlayer(config)
    kani = KaniModel(config, player)
    return config, player, kani


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG, PLAYER, KANI_MODEL = _init_models()
NORMALIZER = VietnameseTTSNormalizer()
SAMPLE_RATE = 22050


def _save_audio(audio: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def _run_standard(text: str, speaker_id: Optional[str]) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    audio, _ = KANI_MODEL.run_model(text, speaker_id=speaker_id)
    elapsed = time.perf_counter() - start
    return audio, elapsed


def _split_text_by_punctuation(text: str, max_chunk_len: int) -> List[str]:
    """
    Tách văn bản thành các đoạn nhỏ dựa trên dấu câu.
    Ưu tiên ngắt theo . ! ? ; : … Sau đó gom lại sao cho mỗi đoạn <= max_chunk_len.
    Nếu vẫn quá dài (ít dấu câu), fallback chia theo độ dài cố định.
    """
    text = text.strip()
    if not text:
        return []

    # Tách sơ bộ theo câu, giữ lại dấu câu ở cuối câu
    # Ví dụ: "Xin chào. Bạn khỏe không?" -> ["Xin chào.", "Bạn khỏe không?"]
    sentence_end_re = re.compile(r"([^.!?;:…]+[.!?;:…]|\S+\s*$)", re.UNICODE)
    sentences = [m.group(0).strip() for m in sentence_end_re.finditer(text)]

    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if not sent:
            continue

        # Nếu câu đơn đã dài hơn max_chunk_len thì cắt cứng theo độ dài
        if len(sent) > max_chunk_len:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sent), max_chunk_len):
                sub = sent[i : i + max_chunk_len].strip()
                if sub:
                    chunks.append(sub)
            continue

        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chunk_len:
            current = f"{current} {sent}"
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return chunks


# --- Main synthesis ---
def synthesize(text: str, speaker_label: str, normalize: bool = True):
    text = (text or "").strip()
    if not text:
        yield None, "⚠️ Vui lòng nhập nội dung.", None
        return

    if len(text) > MAX_TEXT_LEN:
        yield None, f"⚠️ Văn bản quá dài ({len(text)} ký tự). Giới hạn là {MAX_TEXT_LEN} ký tự.", None
        return

    speaker_id = dict(SPEAKER_CHOICES).get(speaker_label, None)

    # --- mô phỏng tiến trình ---
    yield None, "⏳ Đang xử lý văn bản...", None
    time.sleep(0.8)

    # Tách văn bản thành các đoạn theo dấu câu
    raw_chunks = _split_text_by_punctuation(text, MAX_CHARS_PER_CHUNK)
    if not raw_chunks:
        yield None, "⚠️ Không tìm thấy nội dung hợp lệ sau khi xử lý.", None
        return

    if len(raw_chunks) == 1:
        status_msg = "🎧 Đang tạo giọng nói (1 đoạn)..."
    else:
        status_msg = f"🎧 Đang tạo giọng nói ({len(raw_chunks)} đoạn)..."

    yield None, status_msg, None
    time.sleep(0.5)

    audios = []
    total_elapsed = 0.0

    try:
        for idx, chunk in enumerate(raw_chunks, start=1):
            chunk_text = NORMALIZER.normalize(chunk) if normalize else chunk
            audio, elapsed = _run_standard(chunk_text, speaker_id)
            total_elapsed += elapsed

            if audio is None or len(audio) == 0:
                yield None, f"⚠️ Không tạo được audio cho đoạn {idx}.", None
                return

            audios.append(audio)

    except Exception as exc:
        yield None, f"❌ Lỗi khi suy luận: {exc}", None
        return

    if not audios:
        yield None, "⚠️ Không tạo được audio đầu ra.", None
        return

    # Ghép các đoạn audio liên tiếp
    audio_full = np.concatenate(audios)
    wav_path = _save_audio(audio_full)
    duration = len(audio_full) / SAMPLE_RATE
    status = (
        f"✅ Hoàn tất sau {total_elapsed:.2f}s | "
        f"Độ dài audio: {duration:.1f}s | Số đoạn: {len(raw_chunks)}"
    )
    yield wav_path, status, wav_path


# --- Build simple Gradio UI ---
def build_interface():
    examples = [
        ["Khoa – Nam miền Bắc", "Cũng trong thập niên 1960, Jones quyết định đương đầu với một thử thách mới, viết nhạc phim."],
        ["Hùng – Nam miền Nam", "Ông biết hiện giờ nhiều người không còn thích đọc sách nữa, thế nên dù ai đó chỉ vô tình ghé hiệu sách, ông cũng đều trân trọng cả."],
        ["Trinh – Nữ miền Nam", "Đi vào chi tiết Làm việc nhóm và tính cứng nhắc cá nhân là hai điều không thể nào tương thích với nhau."],
        ["David – English (British)", "Ngược lại, những người không thể đào tạo sẽ gặp khó khăn với sự thay đổi và kết quả là họ không thể thích nghi."],
        ["Katie – English (Irish)", "Những người này sẽ vò đầu bứt tai, chịu đựng nỗi đau thể chất khi nghĩ đến chuyện làm những điều khác biệt."],
    ]

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
        gr.Markdown(
            """
            # 😻 Kani TTS Vie – Simple Demo  
            Mô hình tổng hợp giọng nói nhanh và biểu cảm, hỗ trợ tiếng Việt  
            
            💡 *Gradio chưa hỗ trợ streaming trực tiếp. Chế độ này chỉ mô phỏng, nếu muốn streaming thật, tham khảo phiên bản mã nguồn tại https://github.com/pnnbao97/Kani-TTS-Vie.*

            """
        )

        text_input = gr.Textbox(
            label=f"📝 Nội dung (tối đa {MAX_TEXT_LEN} ký tự)",
            placeholder="Nhập văn bản cần chuyển thành giọng nói...",
            lines=6,
            value=(
                "Khi bạn kề vai sát cánh cùng đồng đội của mình, "
                "bạn có thể làm nên những điều phi thường."
            ),
        )

        speaker_dropdown = gr.Dropdown(
            label="🎤 Chọn giọng đọc",
            choices=[label for label, _ in SPEAKER_CHOICES],
            value="Hùng – Nam miền Nam",
        )

        run_button = gr.Button("🎵 Tạo giọng nói", variant="primary")
        status_output = gr.Markdown(label="Trạng thái")
        audio_output = gr.Audio(label="🔊 Kết quả", autoplay=False)
        download_output = gr.File(label="💾 Tải WAV")

        run_button.click(
            fn=synthesize,
            inputs=[text_input, speaker_dropdown],
            outputs=[audio_output, status_output, download_output],
        )

        gr.Examples(
            examples=examples,
            inputs=[speaker_dropdown, text_input],
            label="📚 Ví dụ nhanh"
        )

    demo.queue()
    return demo


demo = build_interface()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),share=True)



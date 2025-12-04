import os
import tempfile
import time
from typing import Optional, Tuple

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
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path

def _run_standard(text: str, speaker_id: Optional[str]) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    audio, _ = KANI_MODEL.run_model(text, speaker_id=speaker_id)
    elapsed = time.perf_counter() - start
    return audio, elapsed

# --- Main synthesis ---
def synthesize(text: str, speaker_label: str, normalize: bool = True):
    text = (text or "").strip()
    if not text:
        yield None, "⚠️ Vui lòng nhập nội dung.", None
        return

    if len(text) > 250:
        yield None, f"⚠️ Văn bản quá dài ({len(text)} ký tự). Giới hạn là 250 ký tự.", None
        return

    speaker_id = dict(SPEAKER_CHOICES).get(speaker_label, None)
    processed_text = NORMALIZER.normalize(text) if normalize else text

    # --- mô phỏng tiến trình ---
    yield None, "⏳ Đang xử lý văn bản...", None
    time.sleep(0.8)

    yield None, "🎧 Đang tạo giọng nói...", None
    time.sleep(0.8)

    try:
        audio, elapsed = _run_standard(processed_text, speaker_id)
    except Exception as exc:
        yield None, f"❌ Lỗi khi suy luận: {exc}", None
        return

    if audio is None or len(audio) == 0:
        yield None, "⚠️ Không tạo được audio đầu ra.", None
        return

    wav_path = _save_audio(audio)
    duration = len(audio) / SAMPLE_RATE
    status = f"✅ Hoàn tất sau {elapsed:.2f}s | Độ dài audio: {duration:.1f}s"
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
            label="📝 Nội dung (tối đa 250 ký tự)",
            placeholder="Nhập văn bản cần chuyển thành giọng nói...",
            lines=4,
            value="Khi bạn kề vai sát cánh cùng đồng đội của mình, bạn có thể làm nên những điều phi thường.",
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
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860),share=True))


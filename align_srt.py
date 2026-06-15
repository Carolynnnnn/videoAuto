"""
align_srt.py — 基于 Whisper 词级时间戳对齐 SRT 字幕

原理：
  1. 用 Whisper 转录音频，获取每个词的 start/end 时间戳
  2. 对每条 SRT 条目，去除标点后在词序列中匹配对应文字
  3. 字幕起点 = 匹配到的首词 start
     字幕终点 = 匹配到的末词 end + trailing_buffer
  4. 保证相邻条目之间无重叠，但允许有空白间隔

用法：
  python align_srt.py \
    --audio  projects/my_first_video/input/voice_full.mp3 \
    --srt    projects/my_first_video/build/subtitle.srt \
    --out    projects/my_first_video/build/subtitle.srt \
    --model  medium \
    --buffer 0.15
"""

import argparse
import re
import unicodedata
import whisper


def strip_punct(s: str) -> str:
    """去除标点和空白，只保留汉字/字母/数字供匹配。"""
    return re.sub(r"[^\w]", "", s, flags=re.UNICODE).replace("_", "")


def parse_srt(path: str):
    """解析 SRT，返回 [(index, start_s, end_s, text), ...]"""
    entries = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        idx = int(lines[0].strip())
        m = re.match(
            r"(\d+):(\d+):(\d+),(\d+)\s+-->\s+(\d+):(\d+):(\d+),(\d+)", lines[1]
        )
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end   = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        text  = " ".join(lines[2:]).strip()
        entries.append((idx, start, end, text))
    return entries


def srt_ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(entries, path: str):
    lines = []
    for idx, start, end, text in entries:
        lines.append(str(idx))
        lines.append(f"{srt_ts(start)} --> {srt_ts(end)}")
        lines.append(text)
        lines.append("")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def match_segment(words, seg_text: str, search_from: int = 0):
    """
    在 words[search_from:] 中找到与 seg_text 最匹配的连续词段。
    返回 (first_word_idx, last_word_idx) 或 None。

    策略：将 seg_text 去标点后与拼接词字符串做滑动窗口匹配。
    """
    target = strip_punct(seg_text)
    if not target:
        return None

    # 尝试不同窗口大小（词数），找能覆盖 target 的最短窗口
    for window in range(1, min(30, len(words) - search_from + 1)):
        for i in range(search_from, len(words) - window + 1):
            chunk = "".join(strip_punct(w["word"]) for w in words[i:i+window])
            if target in chunk or chunk in target:
                # 微调：确保 chunk 覆盖了 target 的大部分
                overlap = len(set(target) & set(chunk))
                if overlap / max(len(target), 1) >= 0.8:
                    return i, i + window - 1
    return None


def align(audio_path: str, srt_path: str, out_path: str,
          model_name: str = "medium", buffer: float = 0.15):

    print(f"[align] 加载 Whisper {model_name} 模型...")
    model = whisper.load_model(model_name)

    print(f"[align] 转录 {audio_path} ...")
    result = model.transcribe(
        audio_path,
        language="zh",
        word_timestamps=True,
        verbose=False,
    )

    # 提取词级时间戳
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            word = w["word"].strip()
            if word:
                words.append({"word": word, "start": w["start"], "end": w["end"]})

    print(f"[align] 识别到 {len(words)} 个词/字")
    for w in words:
        print(f"  {w['start']:.3f}-{w['end']:.3f}  {w['word']}")

    entries = parse_srt(srt_path)
    print(f"\n[align] 共 {len(entries)} 条 SRT 条目，开始对齐...\n")

    aligned = []
    cursor = 0  # 词序列搜索起点，避免重复匹配

    for idx, orig_start, orig_end, text in entries:
        match = match_segment(words, text, search_from=cursor)
        if match is None:
            print(f"  [{idx}] ⚠ 未匹配到词序列，保留原始时间: {text!r}")
            aligned.append((idx, orig_start, orig_end, text))
            continue

        first_i, last_i = match
        new_start = words[first_i]["start"]
        new_end   = words[last_i]["end"] + buffer
        cursor = first_i + 1  # 下一条从此之后搜索

        delta_s = new_start - orig_start
        delta_e = new_end   - orig_end
        print(f"  [{idx}] {text!r}")
        print(f"        start: {orig_start:.3f} → {new_start:.3f}  ({delta_s:+.3f}s)")
        print(f"        end:   {orig_end:.3f} → {new_end:.3f}  ({delta_e:+.3f}s)")
        aligned.append((idx, new_start, new_end, text))

    # 消除相邻条目重叠（后条目 start 不早于前条目 end）
    for i in range(1, len(aligned)):
        idx, s, e, t = aligned[i]
        prev_end = aligned[i-1][2]
        if s < prev_end:
            s = prev_end
        aligned[i] = (idx, s, e, t)

    write_srt(aligned, out_path)
    print(f"\n[align] 写入 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  required=True)
    parser.add_argument("--srt",    required=True)
    parser.add_argument("--out",    required=True)
    parser.add_argument("--model",  default="medium")
    parser.add_argument("--buffer", type=float, default=0.15,
                        help="尾字后追加的显示缓冲（秒）")
    args = parser.parse_args()
    align(args.audio, args.srt, args.out,
          model_name=args.model, buffer=args.buffer)

# yanfa_v2_4min - 视频项目

## 项目信息

- 创建时间：2026-03-09 13:28:40
- 画幅比例：9:16
- 分辨率：1080x1920

## 目录结构

```
yanfa_v2_4min/
  input/
    script.md          # 口播脚本（必需）
    voice_full.wav     # TTS 语音（必需，或通过 --from-pdf 自动生成）
    source.pdf         # 源 PDF（可选）
  build/
    subtitle.srt       # 字幕时间轴
    manifest.json      # 分段工程清单
    diff.json          # 增量更新记录
  render/
    segments/          # 各段视频
    final.mp4          # 最终视频
  assets/
    library/           # 自有素材库
    generated/         # AI 生成素材
  logs/                # 构建日志
```

## 使用方法

### 全量构建（首次）

```bash
# 从脚本+语音开始
python build.py --project projects/yanfa_v2_4min

# 从 PDF 开始（自动生成脚本和语音）
python build.py --project projects/yanfa_v2_4min --from-pdf input/source.pdf

# 跳过 AI 图片生成（更快）
python build.py --project projects/yanfa_v2_4min --no-ai-image
```

### 增量更新（修改字幕后）

```bash
# 修改 build/subtitle.srt 后执行
python build_incremental.py --project projects/yanfa_v2_4min

# 预览变更（不实际执行）
python build_incremental.py --project projects/yanfa_v2_4min --dry-run
```

## 输入文件说明

1. **script.md**：口播脚本，每行一句话
2. **voice_full.wav/mp3**：整段 TTS 语音
3. **source.pdf**（可选）：源文档，使用 `--from-pdf` 自动处理

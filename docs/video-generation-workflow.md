# VideoAuto 端到端工作流程（文字 -> 成片）

本文描述当前项目从输入文本到最终视频输出的完整链路，包含主流程、可选分支（PDF、Pexels、AI 图片、Pixelle）、产物路径以及失败回退机制。

## 1. 总览

当前流水线可分为两段：

- 前置内容生产（P1-P3）：把文档内容变成可播报脚本与完整旁白音频
- 视频生产主线（S1-S6）：把音频+字幕+视觉计划变成分段视频并合成成片

主流程阶段如下：

1. P1: PDF 文本/图片抽取（可选，若输入已是脚本可跳过）
2. P2: LLM 脚本生成（默认 DeepSeek）
3. P3: TTS 语音生成（默认 Minimax）
4. S1: Whisper 对齐生成 SRT 字幕
5. S2: 生成 Manifest（统一分段与全局风格）
6. S3: 生成 Visual Plan（每段镜头计划）
7. S4: 素材执行（检索/生成/回退）
8. S5: 分段渲染
9. S6: 拼接合成最终视频

## 2. 入口与触发方式

### 2.1 GUI 主入口

- 模块入口：`python3 -m src.gui.app`
- GUI 会按顺序触发 P1-P3 和 S1-S6
- 在 GUI 主流程中，S1 默认采用本地 Whisper（`use_local_whisper=True`）

### 2.2 脚本/函数入口

各阶段在 `src/steps/` 目录下有独立步骤函数，可单独调用用于调试或增量执行。

## 3. 阶段明细（输入 / 输出 / 依赖）

## P1：PDF -> 文本 + 图片抽取（可选）

- 主要实现：`src/steps/step_pdf.py` -> `extract_pdf_text`, `extract_pdf_images`
- 输入：`input/*.pdf`
- 输出：
  - 文本：`extracted/content.md`（或同类路径）
  - 图片：`extracted/images/page-*.png`
- 机制：
  - 优先 `pdftotext`、`pdfimages`
  - 不可用时回退 PyPDF2 / pdf2image

## P2：文本 -> 脚本（LLM）

- 主要实现：`src/steps/step_pdf.py` -> `generate_script_from_text`
- 默认服务：DeepSeek（OpenAI 兼容接口）
- 输入：P1 输出文本（或已有文本）
- 输出：`input/script.md`
- 关键参数：`llm_model`, `api_key`, `base_url`, `max_chars`

## P3：脚本 -> 语音（TTS）

- 主要实现：`src/steps/step_pdf.py` -> `generate_tts_minimax`
- 默认服务：Minimax (Production Policy)
- 输入：`input/script.md`
- 输出：`input/voice_full.mp3`
- 特性：
  - **生产策略**: 生产环境下强制使用 Minimax。ElevenLabs/OpenAI 仅在 `PIXELLE_TEST_MODE=1` 时可用。
  - 长文本自动分段合成
  - 最终自动拼接为单文件音频

## S1：语音/脚本对齐 -> SRT

- 主要实现：`src/steps/step1_align.py` -> `run_step1`
- 输入：
  - 音频：`input/voice_full.mp3`
  - 可选脚本：`input/script.md`
- 输出：`build/subtitle.srt`
- 路由：
  - 本地 Whisper：`transcribe_with_whisper_local`
  - API Whisper：`transcribe_with_whisper_api`
- 后处理：
  - 句子对齐
  - 过短段合并、过长段拆分
  - SRT 校验并写出

## S2：SRT -> Manifest

- 主要实现：`src/steps/step2_manifest.py`（由 GUI 流程调用）
- 输入：`build/subtitle.srt` + 音频
- 输出：`build/manifest.json`
- 作用：
  - 将视频拆成 Segment（含 index/key/text/start/end/duration 等）
  - 写入全局样式（画幅、分辨率、FPS、字幕风格）
- **分段策略 (Segmentation Policy)**:
  - 支持通过 CLI 参数控制：`--min-duration`, `--max-duration`, `--target-min-duration`, `--target-max-duration`。
  - 此外，可通过 `--duration-minutes` (1, 2, 3，默认 1) 指定目标视频时长。
  - 默认 Minimax 视频生成推荐范围：1.5s - 10.0s。

## S3：Segment -> Visual Plan

- 主要实现：`src/steps/step3_visual_plan.py` -> `run_step3`
- 输入：`manifest.json`（包含分段与文本）
- 输出：更新后的 `manifest.json`（每段附加 `visual_plan`）
- 作用：
  - 为每段决定画面类型（如 `pdf_chart` / `broll` / `ai_image` / `kinetic_text` / `template`）
  - 生成关键词、素材提示词、镜头运动、叠加信息
- 特性：
  - 计划缓存（按 plan hash）
  - 增量更新时复用稳定计划

## S4：素材执行（检索/生成/回退）

- 主要实现：`src/steps/step4_assets.py` -> `run_step4`
- 输入：带 `visual_plan` 的 `manifest.json`
- 输出：素材文件 + 更新 `asset_refs` 的 `manifest.json`

### 素材生成模式 (Material Modes)

系统支持三种素材生成模式，通过 `--material-mode` 参数配置：

1. **auto** (默认): 兼容模式。按 PDF -> Pexels -> AI -> 模板的顺序尝试。
2. **ai_preferred**: AI 优先模式。优先尝试 AI 生成路径（Minimax/Pixelle），若失败则回退到 Pexels 或 PDF 素材。
3. **ai_only**: 严格 AI 模式。仅允许 AI 生成路径。禁止使用 `pdf_chart` 和 Pexels 素材。
   - **AI 额度限制**: 每个视频最多允许 6 个 AI 生成片段（硬上限）。
   - **回退机制**: 对于超过 6 个的片段，系统会自动降级到通用模板（Template）以确保流程完成。
   - **失败语义**: 若在 6 个额度内的 AI 生成因 Provider 耗尽或技术故障失败，将返回显式的 `ai_only_exhausted` 状态，不进行非 AI 兜底。

### 素材优先级与路径

在 `auto` 模式下，优先级如下：

1. PDF 图表（命中时直接复用）
2. Pexels 视频（若启用并命中）
3. Pexels 图片（视频未命中时回退）
4. AI 图片生成（如 DALL-E，按配置启用）
5. 通用模板（最终兜底）

### Pixelle 与 AI 路径回退

当启用 AI 路径（`ai_preferred` 或 `ai_only`）时，系统采用以下回退策略：

- **Minimax-First**: 在 `PIXELLE_BACKEND_MODE=direct` 时，优先调用 Minimax 适配器。
- **Legacy-Secondary**: 若 Minimax 调用失败或不可用，系统会尝试回退到旧版 Provider 路径（需配置 `PIXELLE_PROVIDER_API_KEY`）。
- **T2A 契约**: Minimax T2A 接口支持返回 HTTP URL 或 Hex 编码的音频数据，系统会自动识别并处理。
- **失败诊断**: 每次尝试都会记录详细的 `fallback_diagnostic`，包括尝试过的 provider 阶段、错误类别和原因代码。

### 环境变量要求

- `PIXELLE_BACKEND_MODE`: `direct` (Minimax 优先) 或 `legacy` (仅旧版路径)。
- `MINIMAX_API_KEY`: `direct` 模式下必需。
- `PIXELLE_PROVIDER_API_KEY`: `legacy` 模式或 AI 回退路径下必需。

## S5：分段渲染

- 主要实现：`src/steps/step5_render.py`
- 输入：`manifest.json` + 每段素材 + 字幕/样式参数
- 输出：`build/rendered/seg_*.mp4`（或同类分段目录）
- 作用：
  - 把每个 Segment 独立渲染为标准化短片段
  - 应用字幕、基础动效、转场相关配置

## S6：拼接合成最终视频

- 主要实现：`src/steps/step6_concat.py`
- 输入：全部 `seg_*.mp4` + 完整旁白音频（可选 BGM）
- 输出：`build/render/final.mp4`
- 机制：
  - 使用 FFmpeg 拼接视频片段
  - 同步或混合音轨后输出最终成片

## 4. 质量门禁 (Gate Profiles)

系统在渲染完成后会执行连贯性与质量检查，通过 `--gate-profile` 参数控制：

- **release** (默认): 严格模式。任何关键素材缺失或渲染失败都将导致构建中断（Exit 1）。
- **preview**: 预览模式。仅输出警告信息，不中断构建流程，允许生成包含占位符的视频。

## 5. 当前涉及的 API/服务能力（服务商无关）

必需能力（标准主流程）：

- LLM 文本生成（脚本与视觉计划）
- TTS 语音合成（生产环境强制 Minimax）
- ASR/对齐（Whisper 本地或 API）
- 素材检索（Pexels 类图库）

可选增强：

- 文生图（AI 图片兜底）
- 生成式视频（Pixelle: digital_human / i2v / action_transfer）

## 6. 失败回退与稳态策略

- 阶段内回退：
  - PDF 抽取工具缺失时回退 Python 库
  - 素材检索失败时自动降级到下一级来源，最终模板兜底
- 稳态控制（主要在 Step4/Pixelle）：
  - 重试与错误分类
  - 并发上限
  - 熔断
  - 配额控制
  - 灰度/影子执行

## 6. 增量更新模式（提升二次编辑效率）

当脚本/字幕小范围修改时，不必全量重跑：

- 复用未变 Segment 的视觉计划与素材引用
- 仅重算受影响段落的计划、素材与渲染
- 末端重新拼接，快速得到新版本成片

## 7. 关键产物路径速查

- 脚本：`input/script.md`
- 音频：`input/voice_full.mp3`
- 字幕：`build/subtitle.srt`
- 清单：`build/manifest.json`
- 分段视频：`build/rendered/seg_*.mp4`
- 最终视频：`build/render/final.mp4`

## 8. 一句话总结

本项目是“文本驱动”的自动视频流水线：先把文本变成可播报的时间轴（语音+字幕+分段），再为每段自动规划并落地素材，最后渲染分段并拼接成最终短视频。

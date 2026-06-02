# 视频自动化生产工作流 (Video Automation Pipeline)

这是一个功能齐全、模块化的 Python 工作流，旨在实现从 PDF 或文本脚本到最终成片的自动化视频生产。其核心特性是 **Segment 化** 和 **增量更新**，当字幕或脚本发生变更时，系统能智能地只重新渲染受影响的视频片段，从而极大地提升了修改和迭代的效率。

## 核心功能

- **全自动流程**: 从 PDF/脚本输入到 `final.mp4` 输出，全程自动化。
- **增量更新**: 修改字幕后，仅需数秒即可重新合成视频，无需全量渲染。
- **模块化设计**: 每个步骤（音频对齐、素材规划、渲染、合成）都解耦为独立模块，易于扩展和维护。
- **LLM 驱动的视觉规划**: 利用大型语言模型（如 GPT-4）为每句字幕自动生成视觉计划（`VisualPlan`），决定画面类型、关键词和镜头运动。
- **灵活的素材策略**: 支持多种素材来源，并按优先级选用：PDF 内嵌图表 > 自有素材库 > AI 生成图片 > 通用模板背景。
- **可配置的构建参数**: 视频画幅、分辨率、TTS 声音、LLM 模型等均可通过命令行参数或配置文件进行设置。
- **健壮的容错机制**: 在素材生成或渲染失败时，会自动降级到备用方案（如通用模板），确保主流程不被中断。

## 系统架构

工作流遵循“数据驱动”的核心理念，以 `manifest.json` 文件作为生产过程的“唯一事实来源 (SSOT)”。

1.  **输入 (Input)**: 可以是一个 PDF 文档，或者一个写好的口播脚本 (`script.md`) 和对应的 TTS 语音 (`voice_full.wav`)。
2.  **音频对齐 (Alignment)**: 将语音和脚本对齐，生成带精确时间戳的 SRT 字幕文件 (`subtitle.srt`)。
3.  **清单生成 (Manifest Generation)**: 将 SRT 文件解析成 `manifest.json`。此文件将视频拆分为多个 `Segment`，每个 `Segment` 包含文本、时间、ID 等元数据。
4.  **视觉规划 (Visual Planning)**: 对每个 `Segment`，调用 LLM 生成 `VisualPlan`，这是一个结构化的 JSON 对象，描述了该片段的画面构成。
5.  **素材执行 (Asset Execution)**: 根据 `VisualPlan`，系统自动去查找或生成（例如使用 DALL-E 3）所需素材。
6.  **分段渲染 (Segment Rendering)**: 使用 FFmpeg 将每个 `Segment` 的素材、音频片段和字幕文本渲染成独立的短视频 (`seg_xxxx.mp4`)。
7.  **拼接合成 (Concatenation)**: 将所有渲染好的 `Segment` 视频按顺序拼接，并混入完整的旁白音轨和背景音乐，最终输出 `final.mp4`。

### 增量更新机制

当用户修改了字幕并重新生成 SRT 文件后，增量构建脚本 (`build_incremental.py`) 会：

1.  生成一份新的 `manifest.json`。
2.  与旧的 `manifest.json`进行 **Diff**，找出 `added`（新增）、`removed`（删除）、`changed`（变更）和 `unchanged`（未变）的 `Segment`。
3.  对于 `unchanged` 的 `Segment`，直接复用其已有的视频片段和素材。
4.  只对 `added` 和 `changed` 的 `Segment` 重新执行视觉规划、素材获取和渲染步骤。
5.  最后，重新拼接所有视频片段，快速生成最终视频。

## 如何使用

### 环境要求

- Python 3.10+
- FFmpeg
- Poppler (`pdftotext`, `pdfimages`)
- OpenAI API Key (需要配置环境变量 `OPENAI_API_KEY`)
- Pixelle Video AI Keys (见下文配置)

### 配置与模式

项目使用环境变量进行配置。建议将 `.env.example` 复制为 `.env` 并填写相关密钥。

#### 运行模式
- **Local-Test**: `PIXELLE_TEST_MODE=1`。本地确定性测试模式，无需 GPU 或 API 密钥。在此模式下，允许使用所有 TTS 提供商（如 ElevenLabs, OpenAI）进行测试。
- **Minimax-First (direct)**: `PIXELLE_TEST_MODE=0`, `PIXELLE_BACKEND_MODE=direct`。生产模式，直接调用 Minimax 适配器。需配置 `MINIMAX_API_KEY`。**生产环境下仅允许使用 `minimax` 作为 TTS 提供商。**
- **Legacy-Provider (legacy)**: `PIXELLE_TEST_MODE=0`, `PIXELLE_BACKEND_MODE=legacy`。兼容模式，调用旧版 Provider 路径。需配置 `PIXELLE_PROVIDER_API_KEY`。

#### 关键参数与策略
- **TTS 提供商**: 通过 `--tts-provider` 指定。生产模式下强制为 `minimax`。
- **质量门禁 (Gate Profile)**: 通过 `--gate-profile` 控制。
  - `release` (默认): 严格模式，若检测到连贯性问题（如素材缺失、渲染失败）将直接中断构建。
  - `preview`: 预览模式，仅输出警告，不中断流程。
- **分段策略 (Segmentation)**: 支持通过 `--min-duration`, `--max-duration` 等参数精细控制视频分段。此外，可通过 `--duration-minutes` (可选 1, 2, 3，默认 1) 指定目标视频时长。Minimax 视频生成的推荐范围为 1.5s 至 10s。

#### 素材生成模式 (Material Modes)
通过 `--material-mode` 参数控制素材生成的策略：
- **auto** (默认): 兼容模式。按 `PDF 图表 -> Pexels 视频 -> Pexels 图片 -> AI 生成 -> 通用模板` 顺序尝试。
- **ai_preferred**: AI 优先模式。优先尝试 AI 生成（Minimax/Pixelle），失败后回退到 Pexels 或 PDF 素材。
- **ai_only**: 严格 AI 模式。仅允许 AI 生成路径。禁止使用 `pdf_chart` 和 Pexels 素材。
  - **AI 额度限制**: 每个视频最多允许 6 个 AI 生成片段（硬上限）。
  - **回退机制**: 对于超过 6 个的片段，系统会自动降级到通用模板（Template）以确保流程完成。
  - **失败语义**: 若在 6 个额度内的 AI 生成因 Provider 耗尽或技术故障失败，将返回显式的 `ai_only_exhausted` 状态，不进行非 AI 兜底。

#### 关键环境变量
- `PIXELLE_BACKEND_MODE`: 设置为 `direct` (默认) 使用 Minimax 优先路径，或 `legacy` 使用旧版路径。
- `MINIMAX_API_KEY`: Minimax API 密钥（`direct` 模式必需）。
- `PIXELLE_PROVIDER_API_KEY`: Legacy Provider API 密钥（`legacy` 模式必需）。
- `PIXELLE_CONCURRENCY_LIMIT`: 并发限制（默认 4）。

### 生产预检 (Preflight Runbook)

在正式生产运行前，请务必执行以下预检流程：

1. **环境验证**: 运行配置验证脚本，确保所有必要的 API 密钥和环境变量已正确配置。
   ```bash
   python3 pixelle_snapshot/validate_config_profiles.py
   ```
2. **架构审计**: 确保没有违反模块边界的导入（如在核心逻辑中引入了 UI 框架）。
   ```bash
   python3 pixelle_snapshot/audit_boundaries.py
   ```
3. **空运行 (Dry-run)**: 使用 `--dry-run` 参数进行一次完整的逻辑扫描，不执行实际的 AI 生成。
   ```bash
   PIXELLE_TEST_MODE=1 python3 build_incremental.py --project projects/demo --dry-run
   ```

更多详细信息请参考 `pixelle_snapshot/PRODUCTION_RUNBOOK.md` 和 `pixelle_snapshot/TROUBLESHOOTING_RUNBOOK.md`。

### 安装依赖

```bash
sudo pip3 install -r requirements.txt
```
*(注：`requirements.txt` 文件尚未创建，但已包含必要的库如 `openai`, `openai-whisper`, `Pillow` 等)*

### 1. 初始化项目

首先，使用 `init_project.py` 脚本创建一个新的项目目录。

```bash
# 创建一个名为 'my_first_video' 的项目，画幅为 9:16
python3.11 init_project.py --name my_first_video --base-dir ./projects
```

这会生成一个包含标准目录结构和配置文件的项目文件夹 `projects/my_first_video`。

### 2. 准备输入文件

进入项目 `input` 目录 (`projects/my_first_video/input/`)，准备以下文件：

- **`script.md`**: 你的口播脚本，每句话占一行。
- **`voice_full.wav` 或 `voice_full.mp3`**: 与脚本内容匹配的完整旁白音频。

如果你想从 PDF 开始，只需将 PDF 文件放入 `input` 目录。

### 3. 执行构建

#### 全量构建 (首次运行)

```bash
# 从脚本和音频开始构建
python3.11 build.py --project ./projects/my_first_video

# 指定目标时长为 2 分钟
python3.11 build.py --project ./projects/my_first_video --duration-minutes 2

# 或者，从 PDF 开始全自动构建（将自动生成脚本和语音）
python3.11 build.py --project ./projects/my_first_video --from-pdf ./projects/my_first_video/input/your_doc.pdf

# 使用本地 Whisper 模型进行语音转写（更快，但可能需要 GPU）
python3.11 build.py --project ./projects/my_first_video --local-whisper
```

构建成功后，最终视频会出现在 `render/final.mp4`。

#### 增量更新 (修改字幕后)

1.  打开 `build/subtitle.srt` 文件，手动修改你想要的字幕文本或时间。
2.  运行增量构建脚本：

```bash
python3.11 build_incremental.py --project ./projects/my_first_video
```

系统将自动检测变更并只重新渲染必要的片段，然后在几秒钟内完成最终视频的合成。

## 目录结构

```
video_pipeline/
├── build.py                # 全量构建入口
├── build_incremental.py    # 增量构建入口
├── init_project.py         # 项目初始化脚本
├── README.md               # 本文档
├── projects/
│   └── demo/               # 示例项目
│       ├── input/
│       ├── build/
│       ├── render/
│       └── ...
└── src/
    ├── core/               # 核心模型、配置、Diff引擎
    ├── steps/              # 各构建步骤的实现
    └── utils/              # 日志、SRT处理等工具
```

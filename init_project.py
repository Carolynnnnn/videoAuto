#!/usr/bin/env python3
"""
项目初始化脚本：init_project.py

创建新项目目录结构，生成默认配置文件。

用法：
  python init_project.py --name my_video --aspect-ratio 9:16
  python init_project.py --name demo --aspect-ratio 3:4
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import ProjectConfig
from src.core.models import GlobalStyle
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="初始化新视频项目")
    parser.add_argument("--name", "-n", required=True, help="项目名称")
    parser.add_argument(
        "--base-dir",
        default="./projects",
        help="项目基础目录（默认 ./projects）",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="9:16",
        choices=["9:16", "3:4", "16:9", "1:1"],
        help="视频画幅（默认 9:16）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("init_project")

    project_root = str(Path(args.base_dir) / args.name)
    cfg = ProjectConfig(project_root=project_root, project_id=args.name)
    cfg.ensure_dirs()

    # 分辨率映射
    resolution_map = {
        "9:16": (1080, 1920),
        "3:4": (1080, 1440),
        "16:9": (1920, 1080),
        "1:1": (1080, 1080),
    }
    w, h = resolution_map[args.aspect_ratio]

    # 生成默认配置文件
    config = {
        "project_id": args.name,
        "created_at": datetime.now().isoformat(),
        "global_style": {
            "subtitle_style": "clean_white",
            "motion_preset": "soft_kenburns",
            "aspect_ratio": args.aspect_ratio,
            "resolution": f"{w}x{h}",
            "fps": 30,
            "font_size": 48,
            "font_color": "white",
            "subtitle_bg": True,
            "subtitle_bg_color": "black@0.5",
            "style_version": "v1",
        },
        "build_params": {
            "min_segment_duration": 1.5,
            "max_segment_duration": 8.0,
            "target_segment_duration": 4.5,
            "time_change_threshold": 0.2,
            "max_retries": 3,
            "tts_voice": "alloy",
            "tts_speed": 1.0,
            "llm_model": "gpt-4.1-mini",
        },
        "paths": {
            "project_root": project_root,
            "input_dir": str(cfg.input_dir),
            "build_dir": str(cfg.build_dir),
            "render_dir": str(cfg.render_dir),
        },
    }

    config_path = Path(project_root) / "project.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成示例 script.md
    example_script = """# 示例脚本

请将此文件替换为您的口播脚本。

每行一句话，建议每句 10-25 字。

示例：
春节红包大战已经落下帷幕。
微信、支付宝、抖音三大平台激烈角逐。
今年的数据有哪些亮点？
让我们一起来看看。
"""
    script_path = cfg.input_dir / "script.md"
    if not script_path.exists():
        script_path.write_text(example_script, encoding="utf-8")

    # 生成 README
    readme = f"""# {args.name} - 视频项目

## 项目信息

- 创建时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 画幅比例：{args.aspect_ratio}
- 分辨率：{w}x{h}

## 目录结构

```
{args.name}/
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
python build.py --project {project_root}

# 从 PDF 开始（自动生成脚本和语音）
python build.py --project {project_root} --from-pdf input/source.pdf

# 跳过 AI 图片生成（更快）
python build.py --project {project_root} --no-ai-image
```

### 增量更新（修改字幕后）

```bash
# 修改 build/subtitle.srt 后执行
python build_incremental.py --project {project_root}

# 预览变更（不实际执行）
python build_incremental.py --project {project_root} --dry-run
```

## 输入文件说明

1. **script.md**：口播脚本，每行一句话
2. **voice_full.wav/mp3**：整段 TTS 语音
3. **source.pdf**（可选）：源文档，使用 `--from-pdf` 自动处理
"""
    readme_path = Path(project_root) / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    logger.info(f"✅ 项目初始化完成: {project_root}")
    logger.info(f"  配置文件: {config_path}")
    logger.info(f"  示例脚本: {script_path}")
    logger.info(f"\n下一步：")
    logger.info(f"  1. 将 voice_full.wav 放入 {cfg.input_dir}")
    logger.info(f"  2. 编辑 {script_path}")
    logger.info(f"  3. 运行: python build.py --project {project_root}")


if __name__ == "__main__":
    main()

"""
创建测试用 PDF 文件：《人工智能改变未来》
使用 reportlab 生成内容丰富的 PDF，用于端到端测试
"""
import sys
from pathlib import Path

def create_test_pdf(output_path: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import subprocess, os

    # 尝试注册中文字体
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    cn_font = "Helvetica"
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("CJK", fp))
                cn_font = "CJK"
                print(f"  已注册中文字体: {fp}")
                break
            except Exception as e:
                print(f"  字体注册失败 {fp}: {e}")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    def make_style(name, parent="Normal", fontSize=12, leading=18,
                   textColor=colors.black, spaceAfter=6, alignment=0, bold=False):
        return ParagraphStyle(
            name,
            parent=styles[parent],
            fontName=cn_font,
            fontSize=fontSize,
            leading=leading,
            textColor=textColor,
            spaceAfter=spaceAfter,
            alignment=alignment,
        )

    title_style = make_style("Title", fontSize=24, leading=32,
                              textColor=colors.HexColor("#1a1a2e"), spaceAfter=12,
                              alignment=1, bold=True)
    subtitle_style = make_style("Subtitle", fontSize=14, leading=20,
                                 textColor=colors.HexColor("#16213e"), spaceAfter=20,
                                 alignment=1)
    h1_style = make_style("H1", fontSize=18, leading=26,
                           textColor=colors.HexColor("#0f3460"), spaceAfter=10, bold=True)
    h2_style = make_style("H2", fontSize=14, leading=22,
                           textColor=colors.HexColor("#533483"), spaceAfter=8, bold=True)
    body_style = make_style("Body", fontSize=11, leading=18,
                             textColor=colors.HexColor("#333333"), spaceAfter=8)
    highlight_style = make_style("Highlight", fontSize=11, leading=18,
                                  textColor=colors.HexColor("#e94560"), spaceAfter=8)

    story = []

    # ── 封面 ──
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("人工智能改变未来", title_style))
    story.append(Paragraph("2024 年 AI 技术发展白皮书", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("发布机构：未来科技研究院", body_style))
    story.append(Paragraph("发布日期：2024 年 12 月", body_style))
    story.append(PageBreak())

    # ── 第一章：AI 概述 ──
    story.append(Paragraph("第一章：人工智能的崛起", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1.1 什么是人工智能？", h2_style))
    story.append(Paragraph(
        "人工智能（Artificial Intelligence，AI）是指由计算机系统展现出的智能行为，"
        "包括学习、推理、问题解决、感知和语言理解等能力。"
        "自 1956 年达特茅斯会议正式提出 AI 概念以来，"
        "这一领域经历了多次技术浪潮，如今已进入以大语言模型为核心的新纪元。",
        body_style
    ))

    story.append(Paragraph("1.2 AI 发展的三次浪潮", h2_style))
    data = [
        ["时期", "核心技术", "代表成果"],
        ["1950-1980年代", "符号主义、专家系统", "IBM 深蓝下棋程序"],
        ["1990-2010年代", "机器学习、神经网络", "人脸识别、语音助手"],
        ["2010年至今", "深度学习、大语言模型", "ChatGPT、Sora、GPT-4"],
    ]
    table = Table(data, colWidths=[4*cm, 6*cm, 6*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), cn_font),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5*cm))

    # ── 第二章：核心技术 ──
    story.append(Paragraph("第二章：核心技术突破", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("2.1 大语言模型（LLM）", h2_style))
    story.append(Paragraph(
        "大语言模型是当前 AI 技术的核心驱动力。"
        "以 GPT-4、Claude、Gemini 为代表的模型，"
        "参数量已突破万亿级别，具备强大的文本理解、生成和推理能力。"
        "2024 年，多模态大模型成为主流，能够同时处理文字、图像、音频和视频。",
        body_style
    ))

    story.append(Paragraph("2.2 AI 生成内容（AIGC）", h2_style))
    story.append(Paragraph(
        "AIGC 技术正在重塑内容创作行业。"
        "文字生成、图像生成、视频生成、音乐创作……"
        "AI 已经能够在各个创意领域媲美甚至超越人类水平。"
        "Sora 的发布标志着 AI 视频生成进入新时代，"
        "一段文字描述即可生成高质量的分钟级视频。",
        body_style
    ))
    story.append(Paragraph(
        "关键数据：2024 年全球 AIGC 市场规模达到 1320 亿美元，"
        "预计 2030 年将突破 1.3 万亿美元。",
        highlight_style
    ))

    story.append(Paragraph("2.3 AI Agent 自主智能体", h2_style))
    story.append(Paragraph(
        "AI Agent 是 2024 年最受关注的技术方向之一。"
        "不同于传统 AI 工具，Agent 能够自主规划、调用工具、执行任务，"
        "实现从「问答机器」到「数字员工」的跨越。"
        "AutoGPT、Devin、Manus 等产品的出现，"
        "让 AI 自主完成复杂工作流成为现实。",
        body_style
    ))

    story.append(PageBreak())

    # ── 第三章：行业应用 ──
    story.append(Paragraph("第三章：行业变革与应用场景", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.3*cm))

    industries = [
        ("医疗健康", "AI 辅助诊断准确率超过专科医生，药物研发周期缩短 60%"),
        ("教育培训", "个性化学习路径，AI 教师 7×24 小时在线辅导"),
        ("金融科技", "智能风控、量化交易、个人理财顾问全面 AI 化"),
        ("内容创作", "AI 写作、绘画、视频制作，创作效率提升 10 倍"),
        ("制造业", "智能工厂、预测性维护、质量检测自动化"),
        ("零售电商", "智能推荐、虚拟试穿、客服机器人全面普及"),
    ]

    for industry, desc in industries:
        story.append(Paragraph(f"▶ {industry}", h2_style))
        story.append(Paragraph(desc, body_style))

    # ── 第四章：未来展望 ──
    story.append(Paragraph("第四章：未来展望与挑战", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("4.1 通用人工智能（AGI）的到来", h2_style))
    story.append(Paragraph(
        "业界普遍预测，AGI 将在 2025-2030 年间实现重大突破。"
        "届时，AI 将具备跨领域的通用推理能力，"
        "能够像人类一样学习任何新技能，解决任何复杂问题。"
        "这将是人类历史上最深刻的技术革命。",
        body_style
    ))

    story.append(Paragraph("4.2 AI 安全与伦理", h2_style))
    story.append(Paragraph(
        "随着 AI 能力的快速提升，安全与伦理问题日益突出。"
        "如何确保 AI 系统的可解释性、公平性和安全性，"
        "如何防范 AI 被滥用于虚假信息传播和网络攻击，"
        "已成为全球政府和科技公司共同面对的核心议题。",
        body_style
    ))

    story.append(Paragraph("4.3 人机协作的新范式", h2_style))
    story.append(Paragraph(
        "未来不是 AI 取代人类，而是人机协作创造更大价值。"
        "掌握 AI 工具的人将拥有超强的生产力，"
        "能够以一人之力完成过去需要团队才能完成的工作。"
        "学会与 AI 协作，是每个人在 AI 时代最重要的核心技能。",
        body_style
    ))

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#e94560")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "结语：AI 时代已经到来，机遇与挑战并存。"
        "拥抱变化，持续学习，才能在这场技术革命中立于不败之地。",
        highlight_style
    ))

    doc.build(story)
    print(f"✓ PDF 已生成: {output_path}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/video_pipeline/test_input.pdf"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    create_test_pdf(output)

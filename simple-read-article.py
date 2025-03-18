#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import requests
import logging
from typing import List, Dict, Any, Tuple
import time
import math
import os
import re
import sys
import argparse  # 添加argparse模块

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("文档翻译")

def extract_text_blocks(input_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """提取并预处理文本块和图像信息

    Args:
        input_path: 输入JSON文件路径

    Returns:
        Tuple[List[Dict], List[Dict]]: 文本块列表和图像块列表

    Raises:
        FileNotFoundError: 输入文件不存在时抛出
        JSONDecodeError: JSON解析失败时抛出
    """
    logger.info("开始提取文本块和图像信息")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_blocks = []
    image_blocks = []
    text_counter = 0
    image_counter = 0

    # 创建图像ID到图像编号的映射
    figure_numbers = {}
    current_figure_number = 1

    # 预处理：查找文本中的Figure引用
    for block in data['blocks']:
        if block['kind']['block_type'] == 'TextBlock' and block['kind']['text'].strip():
            # 查找文本中的Figure X引用
            matches = re.findall(r'Figure\s+(\d+)', block['kind']['text'], re.IGNORECASE)
            for match in matches:
                figure_num = int(match)
                # 找到第一个未分配编号的图像块
                for img_block in data['blocks']:
                    if img_block['kind']['block_type'] == 'Image' and img_block['kind']['id'] not in figure_numbers:
                        figure_numbers[img_block['kind']['id']] = figure_num
                        break

    # 处理所有块
    for block in data['blocks']:
        if block['kind']['block_type'] == 'TextBlock':
            text = block['kind']['text'].strip()
            if text:
                text_counter += 1
                text_blocks.append({
                    "id": f"TB{text_counter:04d}",
                    "original": text,
                    "translation": None
                })
        elif block['kind']['block_type'] == 'Image':
            image_counter += 1

            # 获取图像编号
            figure_number = figure_numbers.get(
                block['kind']['id'],
                current_figure_number
            )

            if block['kind']['id'] not in figure_numbers:
                current_figure_number += 1

            # 创建图像信息
            image_info = {
                "id": f"IMG{image_counter:04d}",
                "image_id": block['kind']['id'],
                "figure_number": figure_number,
                "bbox": block['bbox'],
                "pages_id": block['pages_id'],
                "image_path": f"figures/img_{block['kind']['id']}.png",
                "caption": block['kind'].get('caption')
            }

            image_blocks.append(image_info)

    logger.info(f"共提取 {len(text_blocks)} 个有效文本块")
    logger.info(f"共提取 {len(image_blocks)} 个图像块")
    return text_blocks, image_blocks

def estimate_tokens(text: str) -> int:
    """估计文本的token数量"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text)
    english_chars = total_chars - chinese_chars

    # 中文字符按1.5倍计算，英文按5个字符一个token计算
    tokens = chinese_chars * 1.5 + english_chars / 5

    return math.ceil(tokens)

def should_translate(text: str) -> bool:
    """判断文本是否需要翻译

    Args:
        text: 需要检查的原始文本，长度应大于200字符

    Returns:
        bool: True表示需要翻译，False表示跳过

    Examples:
        >>> should_translate("This is a long English text")
        True
        >>> should_translate("短文本")
        False
        >>> should_translate("123.45 (67%)")
        False
    """
    # 跳过包含中文字符的文本
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return False

    # 跳过短文本（小于200字符）
    if len(text.strip()) < 200:
        return False

    reference_pattern = re.compile(
        r'^\(?\d{1,4}\)?[\s.]+[A-Za-z]+,.*\d{4}.*\d+[-−–]\d+',  # 匹配文献列表
        flags=re.ASCII
    )
    # print(text)
    # print(reference_pattern.match(text))

    return not reference_pattern.match(text)

def translate_batch_with_llm(batch: List[Dict], api_key: str, batch_index: int, total_batches: int) -> List[Dict]:
    """翻译一批文本块
    
    Args:
        batch: 待翻译的文本块批次
        api_key: Gemini API密钥（从环境变量GEMINI_API_KEY中获取）
        batch_index: 当前批次索引
        total_batches: 总批次数
        
    Returns:
        List[Dict]: 翻译后的批次
    """
    logger.info(f"开始翻译批次 {batch_index+1}/{total_batches}，包含 {len(batch)} 个文本块")

    prompt = """请将以下JSON中的文本翻译缩写为中文：
1. 字符数≥200的文本用中文压缩到原文的 20%-40% 左右.
2. 难翻译的部分, 如公式/化学式/专业术语/文献引用/技术术语等保留原样.
3. 文献(REFERENCES) 部分不要翻译.
4. 直接返回JSON格式，保持id和original不变，只修改translation字段.
"""

    input_json = json.dumps(batch, ensure_ascii=False)

    # 请求Gemini API
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt + "\n" + input_json}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8000,
            "topP": 0.9
        }
    }

    # Gemini API端点
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    model = "gemini-2.0-flash-exp"
    endpoint = f"{base_url}/models/{model}:generateContent"
    url = f"{endpoint}?key={api_key}"

    headers = {"Content-Type": "application/json"}
    max_retries = 2

    for attempt in range(max_retries):
        try:
            logger.info(f"批次 {batch_index+1} 第 {attempt+1} 次尝试请求API")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=180
            )
            response.raise_for_status()

            result = response.json()
            content = ""

            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            content += part['text']

                    logger.info(f"批次 {batch_index+1} 翻译成功，尝试提取JSON")

                    try:
                        # 提取JSON内容
                        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if match:
                            # 尝试解析代码块中的JSON
                            json_content = match.group(1).strip()
                        else:
                            # 直接尝试解析全部内容
                            json_content = content.strip()

                        # 清理可能的前导和尾随文本
                        json_content = re.sub(r'^[^{\[]+', '', json_content)
                        json_content = re.sub(r'[^}\]]+$', '', json_content)

                        # 解析JSON并验证
                        translated_batch = json.loads(json_content)
                        
                        # 验证翻译结果
                        if validate_translation_results(batch, translated_batch):
                            logger.info(f"批次 {batch_index+1} 翻译结果有效")
                            return translated_batch
                        else:
                            logger.warning(f"批次 {batch_index+1} 翻译结果无效，重试...")
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON解析错误: {je}")
                        logger.debug(f"尝试解析的内容: {json_content}")
                    except Exception as e:
                        logger.error(f"处理API响应时出错: {e}")
                        logger.debug(f"API响应内容: {content[:500]}...")
            else:
                logger.error(f"API响应缺少候选结果: {result}")

        except requests.exceptions.RequestException as e:
            logger.error(f"API请求错误: {e}")

        # 最后一次尝试失败，等待更长时间再重试
        if attempt < max_retries - 1:
            wait_time = 5 * (attempt + 1)
            logger.info(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

    # 所有尝试都失败，返回原始批次
    logger.error(f"批次 {batch_index+1} 翻译失败，所有重试均未成功，返回未翻译的批次")
    return batch

def clean_markdown(content: str) -> str:
    """清理Markdown格式，提取纯JSON内容"""
    # 已经是JSON数组的情况
    if content.strip().startswith('[') and content.strip().endswith(']'):
        try:
            json.loads(content.strip())
            return content.strip()
        except json.JSONDecodeError:
            pass

    # 提取代码块
    if "```" in content:
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, content)
        if matches:
            return max(matches, key=len).strip()

    # 提取JSON数组
    if '[{' in content or (('[' in content) and ('"id"' in content)):
        array_pattern = r'(\[\s*\{\s*"id"\s*:.*\}\s*\])'
        matches = re.findall(array_pattern, content, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    return content.strip()

def validate_translation_results(original_batch: List[Dict], translated_data: List[Dict]) -> bool:
    """验证翻译结果的完整性和正确性"""
    # 数量不一致
    if len(original_batch) != len(translated_data):
        return False

    # ID不一致
    original_ids = {item['id'] for item in original_batch}
    translated_ids = {item['id'] for item in translated_data}
    if original_ids != translated_ids:
        return False

    # 有缺失的翻译
    missing_translations = sum(1 for item in translated_data
                             if 'translation' not in item
                             or item['translation'] is None
                             or item['translation'] == "")

    # 超过20%缺失则认为不合格
    if missing_translations > len(translated_data) * 0.2:
        return False

    return True

def translate_with_llm(text_blocks: List[Dict], api_key: str, batch_size: int = 20) -> List[Dict]:
    """使用LLM翻译文本块

    Args:
        text_blocks: 待翻译的文本块列表
        api_key: Gemini API密钥（从环境变量GEMINI_API_KEY中获取）
        batch_size: 每批处理的文本块数量

    Returns:
        List[Dict]: 翻译后的文本块列表
    """
    logger.info(f"开始处理翻译任务，共 {len(text_blocks)} 个文本块，批次大小: {batch_size}")
    
    # 先过滤出需要翻译的文本块和不需要翻译的文本块
    to_translate = []
    skip_blocks = []
    
    for block in text_blocks:
        if should_translate(block["original"]):
            to_translate.append(block)
        else:
            # 对不需要翻译的文本块，保持translation为None
            skip_blocks.append(block)
    
    logger.info(f"共有 {len(to_translate)} 个文本块需要翻译，{len(skip_blocks)} 个文本块不需要翻译")
    
    # 如果没有需要翻译的文本块，直接返回原始文本块
    if not to_translate:
        logger.info("没有需要翻译的内容，跳过API调用")
        return text_blocks
    
    # 只对需要翻译的文本块进行批次划分
    total_blocks = len(to_translate)
    num_batches = math.ceil(total_blocks / batch_size)
    logger.info(f"将翻译任务分为 {num_batches} 个批次处理")
    
    # 存储所有翻译结果
    translated_blocks = []
    
    # 按批次处理翻译
    for batch_index in range(num_batches):
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, total_blocks)
        current_batch = to_translate[start_idx:end_idx]
        
        # 翻译当前批次
        translated_batch = translate_batch_with_llm(current_batch, api_key, batch_index, num_batches)
        translated_blocks.extend(translated_batch)
        
        # 休眠一段时间，避免API请求过于频繁
        if batch_index < num_batches - 1:
            time.sleep(2)
    
    # 合并翻译结果和不需要翻译的文本块
    result = translated_blocks + skip_blocks
    
    # 按原始顺序排序（假设每个块有唯一ID）
    id_to_index = {block["id"]: i for i, block in enumerate(text_blocks)}
    result.sort(key=lambda x: id_to_index.get(x["id"], 0))
    
    return result

def generate_html(original_json: Dict, translated_data: List[Dict], output_path: str, image_data: List[Dict] = None):
    """生成最终HTML文件，包含文本和图像信息"""
    logger.info("开始生成HTML文档")

    # 创建翻译映射字典
    translation_map = {item['id']: item for item in translated_data}

    # 检查figures目录是否存在
    figures_dir = "figures"
    has_figures = os.path.isdir(figures_dir)
    if not has_figures:
        logger.warning(f"未找到图像目录: {figures_dir}，将使用占位符代替")

    # 提取Title块并按level构建大纲
    outline_blocks = []
    for block in original_json['blocks']:
        if block['kind']['block_type'] == 'Title' and 'level' in block['kind'] and 'text' in block['kind'] and block['kind']['text']:
            outline_blocks.append({
                'id': block['id'],
                'level': block['kind']['level'],
                'text': block['kind']['text']
            })

    # 按文档中出现的顺序排序
    outline_blocks.sort(key=lambda x: x['id'])

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Processed Document</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 850px; margin: 0 auto; padding: 20px; }
        p { line-height: 1.6; margin: 1em 0; }
        .translated { color: #2c3e50; border-left: 3px solid #3498db; padding-left: 1em; }
        .translated:hover { background-color: #f8f9fa; }
        .original { display: none; color: #7f8c8d; border-left: 3px solid #e74c3c; padding-left: 1em; }
        .original:hover { background-color: #f8f9fa; }
        .image-container {
            margin: 30px 0;
            text-align: center;
            border: 1px solid #eaeaea;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .image-placeholder {
            background-color: #f0f0f0;
            margin: 0 auto;
            display: flex;
            justify-content: center;
            align-items: center;
            max-width: 100%;
            max-height: 300px;
        }
        .image-content {
            max-width: 100%;
            max-height: 600px;
            margin: 0 auto;
            display: block;
            border-radius: 3px;
            transition: transform 0.3s ease;
        }
        .image-content:hover {
            transform: scale(1.02);
        }
        .image-caption {
            font-style: italic;
            color: #555;
            margin-top: 15px;
            padding: 5px 10px;
            background-color: #f9f9f9;
            border-radius: 3px;
        }
        .image-info {
            font-size: 0.8em;
            color: #777;
            margin-top: 8px;
            padding: 0 10px;
        }
        .image-type {
            font-weight: bold;
            color: #3498db;
            margin-bottom: 10px;
            padding: 5px;
            background-color: #f1f8fe;
            border-radius: 3px;
            display: inline-block;
        }
        .figure-number {
            font-weight: bold;
            font-size: 1.2em;
            color: #333;
            margin-bottom: 15px;
            padding: 5px 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            display: inline-block;
            border-bottom: 2px solid #3498db;
        }
        .image-missing {
            color: #d35400;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }

        /* 页眉导航和控制按钮 */
        .header {
            position: sticky;
            top: 0;
            background-color: white;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
            margin-bottom: 20px;
            z-index: 100;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .toggle-btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.2s ease;
        }
        .toggle-btn:hover {
            background-color: #2980b9;
        }
        .title {
            font-size: 1.5em;
            margin: 0;
            color: #34495e;
        }
        .toggle-all-btn {
            background-color: #2ecc71;
        }
        .toggle-all-btn:hover {
            background-color: #27ae60;
        }

        /* 图片查看器模式 */
        .fullscreen-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .fullscreen-image {
            max-width: 90%;
            max-height: 80vh;
            object-fit: contain;
        }
        .fullscreen-caption {
            color: white;
            padding: 10px;
            font-size: 16px;
            text-align: center;
            max-width: 80%;
        }
        .close-fullscreen {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 30px;
            font-weight: bold;
            cursor: pointer;
        }
        .figure-links {
            margin: 15px 0;
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .figure-link {
            padding: 5px 10px;
            background-color: #f1f8fe;
            border-radius: 4px;
            text-decoration: none;
            color: #3498db;
            font-weight: bold;
            transition: background-color 0.2s ease;
        }
        .figure-link:hover {
            background-color: #e1ebf2;
        }

        /* 文章大纲样式 */
        .outline {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .outline-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 10px;
        }
        .outline-items {
            margin-left: 0;
            padding-left: 0;
            list-style-type: none;
        }
        .outline-level-0 { font-weight: bold; margin-top: 10px; }
        .outline-level-1 { margin-left: 20px; }
        .outline-level-2 { margin-left: 40px; }
        .outline-level-3 { margin-left: 60px; }
        .outline-level-4 { margin-left: 80px; }
        .outline-link {
            text-decoration: none;
            color: #2980b9;
            display: inline-block;
            padding: 3px 0;
            transition: color 0.2s ease;
        }
        .outline-link:hover {
            color: #3498db;
            text-decoration: underline;
        }
        .outline-toggle {
            cursor: pointer;
            color: #7f8c8d;
            font-size: 0.9em;
            padding: 5px;
            border-radius: 3px;
            transition: background-color 0.2s ease;
        }
        .outline-toggle:hover {
            background-color: #eaeaea;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="controls">
            <h1 class="title">文档查看器</h1>
            <div>
                <button class="toggle-btn toggle-all-btn" id="toggleAllBtn">显示所有原文</button>
                <button class="toggle-btn" id="increaseFontBtn">放大字体</button>
                <button class="toggle-btn" id="decreaseFontBtn">缩小字体</button>
            </div>
        </div>
    </div>
"""

    # 添加文章大纲
    if outline_blocks:
        html += """
    <div class="outline">
        <div class="outline-title">文章大纲 <span class="outline-toggle" id="outlineToggle">[隐藏]</span></div>
        <ul class="outline-items" id="outlineItems">
"""
        for block in outline_blocks:
            level_class = f"outline-level-{block['level']}"
            anchor_id = f"title-{block['id']}"
            html += f'            <li class="{level_class}"><a href="#{anchor_id}" class="outline-link">{block["text"]}</a></li>\n'

        html += """
        </ul>
    </div>
"""

    # 图片导航链接
    html += """
    <!-- 图片导航链接 -->
    <div class="figure-links" id="figureLinks">
        <!-- 将由JavaScript填充 -->
    </div>
"""

    translated_count = 0
    original_count = 0
    image_count = 0
    image_loaded_count = 0
    figure_links = []
    # 添加一个集合来跟踪已添加的figure_id，避免重复
    added_figure_ids = set()
    # 添加一个字典来跟踪每个figure_number的计数
    figure_count = {}

    # 创建一个字典，用于确定块在原始JSON中的位置顺序
    block_order = {}
    for i, block in enumerate(original_json['blocks']):
        block_id = block.get('id')
        if block_id is not None:
            block_order[block_id] = i

    # 按原始JSON中的顺序处理块
    blocks_by_id = {block.get('id'): block for block in original_json['blocks']}

    # 按顺序处理所有块
    for block_id, order in sorted(block_order.items(), key=lambda x: x[1]):
        block = blocks_by_id.get(block_id)

        if block['kind']['block_type'] == 'Title' and 'level' in block['kind']:
            # 为标题添加锚点ID
            anchor_id = f"title-{block_id}"
            title_level = block['kind']['level']
            title_tag = f"h{title_level + 1}" if title_level < 6 else "h6"

            html += f'<{title_tag} id="{anchor_id}">{block["kind"]["text"]}</{title_tag}>\n'

        elif block['kind']['block_type'] == 'TextBlock':
            text = block['kind']['text'].strip()
            if text:
                block_id = next((item['id'] for item in translated_data
                               if item['original'] == text), None)

                if block_id and translation_map[block_id]['translation']:
                    trans = translation_map[block_id]['translation']
                    # 添加可切换的原文/译文段落
                    html += f'<div class="text-block" id="block-{block_id}">\n'
                    html += f'  <p class="translated" title="点击切换原文/译文">{trans}</p>\n'
                    html += f'  <p class="original">{text}</p>\n'
                    html += f'</div>\n'
                    translated_count += 1
                else:
                    html += f'<p>{text}</p>\n'
                    original_count += 1

        elif block['kind']['block_type'] == 'Image' and image_data:
            # 查找对应的图像信息
            image_info = next((img for img in image_data if img['image_id'] == block['kind']['id']), None)
            if image_info:
                # 计算图像的相对大小
                bbox = block['bbox']
                width = bbox['x1'] - bbox['x0']
                height = bbox['y1'] - bbox['y0']

                # 确保有合理的宽高比例
                aspect_ratio = width / height

                # 创建图像锚点
                figure_number = image_info.get('figure_number', image_count + 1)

                # 为相同figure_number的图片创建唯一ID
                if figure_number in figure_count:
                    figure_count[figure_number] += 1
                    unique_id = f"figure-{figure_number}-{figure_count[figure_number]}"
                else:
                    figure_count[figure_number] = 1
                    unique_id = f"figure-{figure_number}-1"

                # 对于导航链接，我们只使用第一个实例
                figure_id = f"figure-{figure_number}-1"
                figure_label = f"Figure {figure_number}"

                # 添加到图像链接列表，但避免重复
                if figure_number not in added_figure_ids:
                    figure_links.append((figure_id, figure_label))
                    added_figure_ids.add(figure_number)

                html += f'<div class="image-container" id="{unique_id}">\n'
                html += f'  <div class="figure-number">{figure_label}</div>\n'
                html += f'  <div class="image-type">图像类型: {block["kind"]["block_type"]} (ID: {image_info["image_id"]})</div>\n'

                # 检查图像文件是否存在
                image_file = image_info.get('image_path', f'figures/img_{image_info["image_id"]}.png')
                if has_figures and os.path.exists(image_file):
                    # 使用实际图像
                    html += f'  <img src="{image_file}" class="image-content" alt="{figure_label}" data-figure="{figure_number}" />\n'
                    image_loaded_count += 1
                else:
                    # 如果找不到图像文件，使用占位符
                    display_width = min(500, width)  # 最大宽度为500像素
                    display_height = display_width / aspect_ratio

                    # 限制高度
                    if display_height > 300:
                        display_height = 300
                        display_width = display_height * aspect_ratio

                    html += f'  <div class="image-placeholder" style="width: {display_width:.0f}px; height: {display_height:.0f}px;">\n'
                    html += f'    [{figure_label} (ID: {image_info["image_id"]}) 未能加载]\n'
                    html += f'  </div>\n'
                    html += f'  <div class="image-missing">图像文件 {image_file} 未找到</div>\n'

                if image_info['caption']:
                    html += f'  <div class="image-caption">{image_info["caption"]}</div>\n'

                pages = ', '.join(map(str, block['pages_id']))
                html += f'  <div class="image-info">图像位置: 第 {pages} 页, 边界框: x0={bbox["x0"]:.2f}, y0={bbox["y0"]:.2f}, x1={bbox["x1"]:.2f}, y1={bbox["y1"]:.2f}</div>\n'
                html += f'</div>\n'
                image_count += 1

    # 添加全屏图像查看器
    html += """
    <!-- 全屏图像查看器 -->
    <div class="fullscreen-overlay" id="fullscreenOverlay">
        <span class="close-fullscreen" id="closeFullscreen">&times;</span>
        <img class="fullscreen-image" id="fullscreenImage" src="" alt="全屏图像" />
        <div class="fullscreen-caption" id="fullscreenCaption"></div>
    </div>
"""

    # 添加JavaScript代码，处理交互功能
    html += """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // 切换单个段落
    document.querySelectorAll('.text-block').forEach(block => {
        const translated = block.querySelector('.translated');
        const original = block.querySelector('.original');

        if (translated && original) {
            translated.addEventListener('click', function() {
                if (original.style.display === 'block') {
                    original.style.display = 'none';
                    translated.style.display = 'block';
                } else {
                    original.style.display = 'block';
                    translated.style.display = 'none';
                }
            });

            original.addEventListener('click', function() {
                original.style.display = 'none';
                translated.style.display = 'block';
            });
        }
    });

    // 切换所有段落
    const toggleAllBtn = document.getElementById('toggleAllBtn');
    let showingOriginal = false;

    if (toggleAllBtn) {
        toggleAllBtn.addEventListener('click', function() {
            const allTranslated = document.querySelectorAll('.translated');
            const allOriginal = document.querySelectorAll('.original');

            if (showingOriginal) {
                allOriginal.forEach(el => el.style.display = 'none');
                allTranslated.forEach(el => el.style.display = 'block');
                toggleAllBtn.textContent = '显示所有原文';
            } else {
                allOriginal.forEach(el => el.style.display = 'block');
                allTranslated.forEach(el => el.style.display = 'none');
                toggleAllBtn.textContent = '显示所有译文';
            }

            showingOriginal = !showingOriginal;
        });
    }

    // 字体大小调整
    const increaseFontBtn = document.getElementById('increaseFontBtn');
    const decreaseFontBtn = document.getElementById('decreaseFontBtn');

    if (increaseFontBtn) {
        increaseFontBtn.addEventListener('click', function() {
            changeFontSize(1);
        });
    }

    if (decreaseFontBtn) {
        decreaseFontBtn.addEventListener('click', function() {
            changeFontSize(-1);
        });
    }

    function changeFontSize(step) {
        const body = document.body;
        const currentSize = parseInt(window.getComputedStyle(body).fontSize);
        body.style.fontSize = (currentSize + step) + 'px';
    }

    // 全屏图像查看
    const overlay = document.getElementById('fullscreenOverlay');
    const fullscreenImage = document.getElementById('fullscreenImage');
    const fullscreenCaption = document.getElementById('fullscreenCaption');
    const closeBtn = document.getElementById('closeFullscreen');

    // 图像点击打开全屏
    document.querySelectorAll('.image-content').forEach(img => {
        img.addEventListener('click', function() {
            fullscreenImage.src = this.src;

            // 查找图像标题
            const container = this.closest('.image-container');
            const figureNumber = this.getAttribute('data-figure');
            const captionEl = container.querySelector('.image-caption');

            if (captionEl) {
                fullscreenCaption.textContent = 'Figure ' + figureNumber + ': ' + captionEl.textContent;
            } else {
                fullscreenCaption.textContent = 'Figure ' + figureNumber;
            }

            overlay.style.display = 'flex';
            document.body.style.overflow = 'hidden'; // 防止背景滚动
        });

        img.style.cursor = 'zoom-in';
    });

    // 关闭全屏查看
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            overlay.style.display = 'none';
            document.body.style.overflow = 'auto'; // 恢复滚动
        });
    }

    // 点击背景关闭全屏
    if (overlay) {
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                overlay.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
    }

    // Escape键关闭全屏
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && overlay.style.display === 'flex') {
            overlay.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });

    // 填充图像导航链接
    const figureLinks = document.getElementById('figureLinks');
    if (figureLinks) {
"""

    # 动态生成图像链接JavaScript代码
    figure_links_js = ""
    for figure_id, figure_label in figure_links:
        figure_links_js += f'        figureLinks.innerHTML += \'<a href="#{figure_id}" class="figure-link">{figure_label}</a>\';\n'

    html += figure_links_js

    html += """
    }

    // 处理大纲折叠/展开
    const outlineToggle = document.getElementById('outlineToggle');
    const outlineItems = document.getElementById('outlineItems');

    if (outlineToggle && outlineItems) {
        outlineToggle.addEventListener('click', function() {
            if (outlineItems.style.display === 'none') {
                outlineItems.style.display = 'block';
                outlineToggle.textContent = '[隐藏]';
            } else {
                outlineItems.style.display = 'none';
                outlineToggle.textContent = '[显示]';
            }
        });
    }
});
</script>
"""

    html += "</body>\n</html>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML文件已生成: {output_path}")
    logger.info(f"统计: {translated_count} 个翻译段落, {original_count} 个原文段落, {image_count} 个图像 (其中 {image_loaded_count} 个成功加载)")

def main():
    """主程序入口"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="处理JSON文件并生成带翻译的HTML查看器",
        epilog="环境变量:\n  GEMINI_API_KEY - Gemini API密钥 (必需)\n  SKIP_TRANSLATION - 设置为'true'可跳过翻译步骤",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 添加参数
    parser.add_argument("input_json", help="输入的JSON文件路径")
    parser.add_argument("-o", "--output", help="输出的HTML文件路径 (默认为 'output.html')", default="output.html")
    parser.add_argument("-s", "--skip-translation", help="跳过翻译步骤，直接生成HTML", action="store_true")
    parser.add_argument("-b", "--batch-size", help="每批翻译的文本块数量 (默认为5)", type=int, default=20)

    # 解析命令行参数
    args = parser.parse_args()

    # 获取环境变量中的翻译设置
    env_skip_translation = os.environ.get("SKIP_TRANSLATION", "False").lower() in ("true", "1", "yes")

    # 命令行参数优先于环境变量
    skip_translation = args.skip_translation or env_skip_translation

    # 1. 提取文本块和图像信息
    try:
        text_blocks, image_blocks = extract_text_blocks(args.input_json)
    except FileNotFoundError:
        logger.error(f"找不到输入文件: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"无法解析JSON文件: {args.input_json}")
        sys.exit(1)

    # 2. 处理翻译
    if skip_translation:
        logger.info("跳过翻译步骤，直接生成HTML")
        translated_blocks = [
            {
                "id": block["id"],
                "original": block["original"],
                "translation": f"[这是{block['id']}的翻译结果]"
            } for block in text_blocks
        ]
    else:
        # 从环境变量中获取API密钥
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("未设置GEMINI_API_KEY环境变量，请设置后重试")
            sys.exit(1)
        
        # 翻译文本块，使用指定的批处理大小
        translated_blocks = translate_with_llm(text_blocks, api_key, args.batch_size)

    # 3. 保存翻译结果
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    translation_file = os.path.join(output_dir, "translated_blocks.json") if output_dir else "translated_blocks.json"
    with open(translation_file, 'w', encoding='utf-8') as f:
        json.dump(translated_blocks, f, ensure_ascii=False, indent=2)

    # 4. 加载原始JSON并生成HTML
    with open(args.input_json, 'r', encoding='utf-8') as f:
        original_json = json.load(f)

    generate_html(original_json, translated_blocks, args.output, image_blocks)
    logger.info(f"处理完成，输出文件: {args.output}")

if __name__ == "__main__":
    main()

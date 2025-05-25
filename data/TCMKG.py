import time
import re
import json
from openai import OpenAI
import os
import csv
import traceback
from dotenv import load_dotenv
load_dotenv()

# --- API 配置 (保持不变) ---
KIMI_API_KEY = os.getenv("MOONSHOT_API_KEY", "sk-your-kimi-api-key")
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-your-deepseek-api-key")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# --- 选择模型 (保持不变) ---
CURRENT_API_KEY = KIMI_API_KEY
CURRENT_BASE_URL = KIMI_BASE_URL
CURRENT_MODEL_NAME = "moonshot-v1-32k" # 长上下文模型可能更好

# CURRENT_API_KEY = DEEPSEEK_API_KEY
# CURRENT_BASE_URL = DEEPSEEK_BASE_URL
# CURRENT_MODEL_NAME = "deepseek-chat"

# --- 文件路径配置 ---
BASE_BOOKS_DIR = "data/src/classicals"  # 存放古籍txt文件的根文件夹 (直接存放txt)
OUTPUT_CSV_FILE = "tcm_KG.csv" # 新的输出文件名
STATUS_FILE = "processing_status_KG.json" # 新的状态文件名
CHUNK_SIZE_THRESHOLD = 8000  # 字符数，可以尝试减小，并考虑增加重叠
CHUNK_OVERLAP = 500          # 文本块之间的重叠字符数
API_CALL_DELAY = int(os.getenv("API_CALL_DELAY", "30")) # 减少延迟以加快处理，但注意API限制
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "15"))

# --- 初始化API客户端 (保持不变) ---
def get_llm_client():
    global CURRENT_API_KEY # 确保可以修改全局变量
    if not CURRENT_API_KEY or "your-" in CURRENT_API_KEY:
        print("Attempting to load API key from environment variables within get_llm_client...")
        if "moonshot" in CURRENT_MODEL_NAME.lower() or "kimi" in CURRENT_MODEL_NAME.lower() :
            CURRENT_API_KEY = os.getenv("MOONSHOT_API_KEY")
        elif "deepseek" in CURRENT_MODEL_NAME.lower():
            CURRENT_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    if not CURRENT_API_KEY or "your-" in CURRENT_API_KEY or CURRENT_API_KEY is None:
         raise ValueError(f"请为 {CURRENT_MODEL_NAME} 设置有效的API Key (MOONSHOT_API_KEY 或 DEEPSEEK_API_KEY 环境变量)")
    client = OpenAI(api_key=CURRENT_API_KEY, base_url=CURRENT_BASE_URL)
    return client

# --- 文本清理 (保持不变) ---
def clean_text_for_llm(text):
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+', '', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[ 　]+', ' ', text)
    return text

# --- Prompt构建 (核心修改) ---
def create_triple_extraction_prompt_v2(text_chunk, book_name, chapter_name, book_id_display, chunk_idx_display):
    # 严格按照论文中提及的10种关系类型（或其合理中文对应）
    # "Belong to Category", "Include Section", "Include Chapter", "Belong to book",
    # "Treatment Plan", "Treat Disease", "Describe Disease", "Treatment Symptom",
    # "Symptoms Present", "Ingredient Use"
    # 我们需要将这些转换为LLM易于理解并用于生成三元组中"predicate"的中文词汇。
    # 同时，某些关系可能需要LLM从上下文中推断主语和宾语的类型。

    target_relations_map = {
        "属于范畴": "Belong to Category (例如：某方剂 属于范畴 扶正剂)",
        "包含章节": "Include Chapter (例如：{书籍名称} 包含章节 {章节名称})", # 书籍名和章节名作为参数传入
        "包含小节": "Include Section (例如：{章节名称} 包含小节 {小节名称})", # 如果能从小节提取
        "属于书籍": "Belong to book (例如：{章节名称} 属于书籍 {书籍名称})", # 书籍名和章节名作为参数传入
        "定义治疗方案": "Treatment Plan (例如：某某病 定义治疗方案 清热解毒法 或 某某症状 定义治疗方案 某某方剂)", # 这个关系比较抽象，需要LLM理解
        "治疗疾病": "Treat Disease (例如：某方剂 治疗疾病 某某病 或 某药材 治疗疾病 某某病)",
        "描述疾病": "Describe Disease (例如：某某病 描述疾病 病因是XXX，症状是YYY)", # Object可能是较长描述
        "治疗症状": "Treatment Symptom (例如：某方剂 治疗症状 某某症状 或 某药材 治疗症状 某某症状)",
        "表现症状": "Symptoms Present (例如：某某病 表现症状 某某症状 或 某某证候 表现症状 某某症状)",
        "使用药材": "Ingredient Use (例如：某方剂 使用药材 某某药材 或 某治法 使用药材 某某药材)"
    }
    # 仅将中文关系名用于提示，括号内为英文原名和解释，帮助我们理解和设计示例
    desired_relations_chinese = list(target_relations_map.keys())

    system_prompt = f"""
你是高级中医文献分析AI。你的任务是从提供的中医古籍文本片段中，严格按照指定的10种关系类型，提取结构化的知识，形式为“(主语, 关系, 客体)”的三元组。

当前处理的文献信息：
书籍名称: {book_name}
书籍ID（用于内部追踪）: {book_id_display}
当前章节名（或主要内容标识）: {chapter_name}
文本块序号: {chunk_idx_display}

请【仅】提取以下指定的关系类型的三元组：
{', '.join(desired_relations_chinese)}

输出格式要求：
请以JSON数组的格式返回提取到的三元组列表。数组中的每个元素都是一个JSON对象，包含以下三个键：
- "subject": 字符串，表示三元组的主语。
- "predicate": 字符串，表示三元组的关系，【必须是上述指定关系之一】。
- "object": 字符串，表示三元组的客体。

重要指令：
1.  严格遵循指定的关系列表：不要提取列表之外的关系。
2.  准确性至上：如果信息不明确或不符合指定关系，请不要提取。
3.  忠于原文：主语和客体应尽可能使用原文中的具体名称或表述。
4.  关系匹配：仔细判断文本内容最符合列表中的哪一种关系。例如，“XX方由A、B、C组成”应提取为 (XX方, 组成, A), (XX方, 组成, B), (XX方, 组成, C)。但“组成”若不在指定列表，则不能提取。根据当前列表，若XX方治疗YY病，且使用了药材A，则可能是 (XX方, 治疗疾病, YY病) 和 (XX方, 使用药材, A)。
5.  “包含章节”和“属于书籍”关系：
    - 如果当前文本块明确提及了它所属的章节和书籍，你可以生成这样的三元组。
    - 例如，当前文本块来自 `{chapter_name}` 章节，这本书是 `{book_name}`，你可以生成：
      `{{"subject": "{book_name}", "predicate": "包含章节", "object": "{chapter_name}"}}`
      `{{"subject": "{chapter_name}", "predicate": "属于书籍", "object": "{book_name}"}}`
    （注意：这种元数据相关的三元组，我们也可以在代码层面根据上下文自动生成，以减轻LLM负担并确保一致性。这里先保留，看LLM效果）
6. 确保主语、谓语、客体均不为空。

输入文本示例（假设当前处理《金匮要略》的“妇人产后病脉证并治第二十一”章节中的一段）：
「产后腹中痛，法当TARGETTING痛，不满五十日，病名产后腹痛。六七日不解，宜服当归生姜羊肉汤。若转筋者，加附子。若痛不止，休作羊肉汤，宜服当归芍药散。」

期望输出JSON数组示例（严格使用指定关系）：
[
    {{"subject": "产后腹中痛", "predicate": "表现症状", "object": "TARGETTING痛"}},
    {{"subject": "产后腹中痛", "predicate": "属于范畴", "object": "产后病"}},
    {{"subject": "产后腹中痛（不满五十日）", "predicate": "治疗疾病", "object": "当归生姜羊肉汤"}},
    {{"subject": "当归生姜羊肉汤", "predicate": "使用药材", "object": "当归"}},
    {{"subject": "当归生姜羊肉汤", "predicate": "使用药材", "object": "生姜"}},
    {{"subject": "当归生姜羊肉汤", "predicate": "使用药材", "object": "羊肉"}},
    {{"subject": "产后腹中痛（伴转筋）", "predicate": "治疗疾病", "object": "当归生姜羊肉汤加附子"}},
    {{"subject": "当归生姜羊肉汤加附子", "predicate": "使用药材", "object": "附子"}},
    {{"subject": "产后腹中痛（痛不止，不宜羊肉汤）", "predicate": "治疗疾病", "object": "当归芍药散"}},
    {{"subject": "《金匮要略》", "predicate": "包含章节", "object": "妇人产后病脉证并治第二十一"}},
    {{"subject": "妇人产后病脉证并治第二十一", "predicate": "属于书籍", "object": "《金匮要略》"}}
]
---
现在，请根据以下文本内容提取三元组：
章节原文片段：
{text_chunk}
---
请仅返回包含三元组列表的JSON数组，不要包含任何其他解释性文字或Markdown标记。
    """
    return system_prompt

# --- 调用LLM API (与之前版本类似，可微调参数) ---
def extract_triples_with_llm_v2(client, text_chunk, book_name, chapter_name, book_id_display, chunk_idx_display):
    system_prompt_content = create_triple_extraction_prompt_v2(text_chunk, book_name, chapter_name, book_id_display, chunk_idx_display)
    max_retries = 3
    response_content = ""

    for attempt in range(max_retries):
        try:
            print(f"      LLM API call for chunk (attempt {attempt + 1}/{max_retries})...")
            completion = client.chat.completions.create(
                model=CURRENT_MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt_content}],
                temperature=0.0, # 极低温度，追求最精确和结构化的输出
                max_tokens=min(int(CHUNK_SIZE_THRESHOLD * 2), 8000), # 允许输出稍长，因为示例比较详细
                response_format={"type": "json_object"},
            )
            response_content = completion.choices[0].message.content
            clean_response = re.sub(r"^```json\s*|\s*```$", "", response_content, flags=re.MULTILINE).strip()
            extracted_data = json.loads(clean_response)

            valid_triples = []
            data_list_to_process = []
            if isinstance(extracted_data, list):
                data_list_to_process = extracted_data
            elif isinstance(extracted_data, dict) and isinstance(list(extracted_data.values())[0], list):
                 data_list_to_process = list(extracted_data.values())[0] # 兼容 {"some_key": [...]}
            else:
                print(f"      Error: LLM response is not a list or a dict containing a list. Type: {type(extracted_data)}")
                print(f"      LLM raw response (cleaned): {clean_response[:500]}")
                return []

            for item in data_list_to_process:
                if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                    s = str(item["subject"]).strip()
                    p = str(item["predicate"]).strip()
                    o = str(item["object"]).strip()
                    # 校验关系是否在我们的目标列表内 (可选，如果prompt够强，LLM应遵守)
                    # if p not in create_triple_extraction_prompt_v2("", "", "", "", "").split("请【仅】提取以下指定的关系类型的三元组：")[1].split("\n")[1].split(", "):
                    #     print(f"        Warning: Extracted predicate '{p}' not in desired list. Skipping triple: ({s}, {p}, {o})")
                    #     continue
                    if s and p and o:
                         valid_triples.append((s, p, o))
                else:
                    print(f"        Warning: Invalid item in LLM response list: {item}")
            print(f"      Successfully extracted {len(valid_triples)} triples.")
            return valid_triples
        # ... (错误处理与之前类似) ...
        except json.JSONDecodeError as e:
            print(f"      Error: LLM response JSON decode error (attempt {attempt + 1}): {e}")
            print(f"      LLM raw response content: {response_content[:1000]}")
        except Exception as e:
            print(f"      Error: Unknown error during LLM API call (attempt {attempt + 1}): {e}")
            traceback.print_exc()
        if attempt < max_retries - 1:
            print(f"      Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            print(f"      Max retries reached. Failed to extract for this chunk.")
    return []

# --- 状态管理 (与之前版本类似) ---
def load_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                status = json.load(f)
                print(f"Resuming from status: File='{status.get('current_file_path', 'None')}', Chunk Index='{status.get('current_chunk_idx', 0)}'")
                return status
        except Exception as e:
            print(f"Warning: Error loading status file '{STATUS_FILE}': {e}. Starting fresh.")
    return {"processed_files_map": {}, "current_file_path": None, "current_chapter_name_being_processed": None, "current_chunk_idx": 0}

def save_status(status):
    try:
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving status file '{STATUS_FILE}': {e}")

# --- 主处理函数 (核心修改) ---
def process_book_file_v2(filepath, book_name, book_id_display, client, status, writer, csv_file_needs_header):
    print(f"\n======================================================")
    print(f"Processing Book: {book_name} (Path: {filepath})")
    print(f"======================================================")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            book_content_full = f.read()
    except Exception as e:
        print(f"  Error reading file {filepath}: {e}")
        status["processed_files_map"][filepath] = f"error_reading: {e}"
        save_status(status)
        return False

    if not book_content_full.strip():
        print(f"  Book {book_name} is empty. Skipping.")
        status["processed_files_map"][filepath] = "empty"
        save_status(status)
        return True

    current_book_triples_count = 0
    
    # 章节划分：使用正则表达式从文本中提取章节标题和内容
    # 假设章节标题以 <...> 包裹，如 <裘氏原叙> 或 <卷之一·某某论>
    # 这个正则表达式会捕获标题和紧随其后的内容直到下一个标题或文件末尾
    chapters = re.split(r'(<[^>]+>)', book_content_full)
    
    parsed_chapters = []
    current_chapter_title = f"{book_name}_引言或无明确章节部分" # 默认给第一部分一个标题
    current_chapter_content = ""

    if not chapters[0].strip() and len(chapters) > 1: # 如果开头是分隔符
        chapters = chapters[1:] # 跳过开头的空字符串

    for i in range(0, len(chapters)):
        part = chapters[i].strip()
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"): # 是一个章节标题
            if current_chapter_content.strip(): # 保存上一个章节的内容
                parsed_chapters.append({"title": current_chapter_title, "content": current_chapter_content.strip()})
            current_chapter_title = part[1:-1].strip() # 去除尖括号并strip
            current_chapter_content = ""
        else: # 是章节内容
            current_chapter_content += part + "\n"
    
    if current_chapter_content.strip(): # 保存最后一个章节的内容
        parsed_chapters.append({"title": current_chapter_title, "content": current_chapter_content.strip()})

    if not parsed_chapters and book_content_full.strip(): # 如果正则没有匹配到章节，则将整个书视为一个章节
        parsed_chapters.append({"title": f"{book_name}_全文", "content": book_content_full.strip()})
        print(f"  Warning: No explicit chapter markers found in '{book_name}'. Treating entire content as one chapter.")
    
    total_chapters_in_book = len(parsed_chapters)
    print(f"  Book '{book_name}' parsed into {total_chapters_in_book} chapters.")

    # 断点续传的章节名和块索引
    start_processing_chapter_name = None
    start_chunk_idx_for_chapter = 0

    if status.get("current_file_path") == filepath:
        start_processing_chapter_name = status.get("current_chapter_name_being_processed")
        start_chunk_idx_for_chapter = status.get("current_chunk_idx", 0)
        if start_processing_chapter_name:
             print(f"  Resuming '{book_name}' from chapter '{start_processing_chapter_name}', chunk index {start_chunk_idx_for_chapter}")
        else:
            print(f"  Resuming '{book_name}' from first chapter, chunk index {start_chunk_idx_for_chapter} (no specific chapter name in status or status implies start of book)")


    for chapter_idx, chapter_data in enumerate(parsed_chapters):
        chapter_name = chapter_data["title"]
        chapter_content_full = chapter_data["content"]

        # 跳过已处理的章节 (如果状态中记录了比当前章节更后的章节，或当前章节但块索引更大)
        if start_processing_chapter_name:
            if chapter_name != start_processing_chapter_name:
                # 检查当前章节是否在应开始的章节之前
                try:
                    # 这需要一个方法来确定章节的顺序，如果parsed_chapters是按顺序的，可以直接比较索引
                    current_parsed_chapter_idx = parsed_chapters.index(chapter_data)
                    start_status_chapter_idx = -1
                    for idx, c_data in enumerate(parsed_chapters):
                        if c_data["title"] == start_processing_chapter_name:
                            start_status_chapter_idx = idx
                            break
                    if start_status_chapter_idx != -1 and current_parsed_chapter_idx < start_status_chapter_idx:
                        print(f"    Skipping chapter '{chapter_name}' (before resume point '{start_processing_chapter_name}')")
                        continue
                except ValueError: # 理论上不应发生，因为chapter_data来自parsed_chapters
                    pass
            # 如果是应开始的章节，但不是从第一个块开始，则在下面的块循环中处理start_chunk_idx_for_chapter

        print(f"    Processing Chapter {chapter_idx + 1}/{total_chapters_in_book}: '{chapter_name}'")
        status["current_file_path"] = filepath
        status["current_chapter_name_being_processed"] = chapter_name
        # status["current_chunk_idx"] 会在块循环中更新
        # save_status(status) # 在进入块循环前保存一次章节状态

        text_chunks = []
        cleaned_chapter_content = clean_text_for_llm(chapter_content_full)
        
        # 分块逻辑，带重叠
        if len(cleaned_chapter_content) > CHUNK_SIZE_THRESHOLD:
            start = 0
            while start < len(cleaned_chapter_content):
                end = min(start + CHUNK_SIZE_THRESHOLD, len(cleaned_chapter_content))
                text_chunks.append(cleaned_chapter_content[start:end])
                if end == len(cleaned_chapter_content):
                    break
                start += (CHUNK_SIZE_THRESHOLD - CHUNK_OVERLAP) # 滑动窗口
                if start >= len(cleaned_chapter_content): # 防止死循环
                    break
        elif cleaned_chapter_content:
            text_chunks.append(cleaned_chapter_content)

        if not text_chunks:
            print(f"      No text chunks for chapter '{chapter_name}'. Skipping chapter.")
            continue
        
        print(f"      Chapter '{chapter_name}' split into {len(text_chunks)} text chunks (overlap: {CHUNK_OVERLAP if len(text_chunks)>1 else 0}).")

        # 确定当前章节应该从哪个块开始处理
        actual_start_chunk_this_chapter = 0
        if chapter_name == start_processing_chapter_name: # 如果是记录中要恢复的章节
            actual_start_chunk_this_chapter = start_chunk_idx_for_chapter
        
        # 在一个章节处理成功或失败后，将 start_processing_chapter_name 置空，
        # 这样后续章节会从它们的第一个块开始
        # 或者在每次外层循环开始时，只对第一个匹配到的章节使用 status 中的 chunk_idx


        for chunk_idx, text_chunk in enumerate(text_chunks):
            if chunk_idx < actual_start_chunk_this_chapter:
                print(f"        Skipping chunk {chunk_idx + 1} in chapter '{chapter_name}' (before resume point)")
                continue

            status["current_chunk_idx"] = chunk_idx # 更新当前处理的块索引
            save_status(status) # 在处理每个块之前保存状态

            print(f"        Processing chunk {chunk_idx + 1}/{len(text_chunks)} for chapter '{chapter_name}'...")
            
            triples_from_chunk = extract_triples_with_llm_v2(client, text_chunk, book_name,
                                                             chapter_name, book_id_display, chunk_idx + 1)
            
            if triples_from_chunk:
                # CSV: Subject, Predicate, Object, SourceBookName, SourceChapterName
                rows_to_write = [(s, p, o, book_name, chapter_name) for s, p, o in triples_from_chunk]
                
                if csv_file_needs_header[0]:
                    writer.writerow(['Subject', 'Predicate', 'Object', 'SourceBookName', 'SourceChapterName'])
                    csv_file_needs_header[0] = False
                writer.writerows(rows_to_write)
                current_book_triples_count += len(rows_to_write)
                print(f"          Chunk {chunk_idx + 1} processed. {len(rows_to_write)} triples extracted and written.")
            else:
                print(f"          No triples extracted from chunk {chunk_idx + 1}.")

            time.sleep(API_CALL_DELAY)
        
        # 当前章节所有块处理完成后，重置下一个章节的起始块索引
        # 并且清除 start_processing_chapter_name，确保后续章节从头开始
        start_processing_chapter_name = None 
        start_chunk_idx_for_chapter = 0
        status["current_chapter_name_being_processed"] = chapter_name # 记录本章已完成（或尝试完成）
        status["current_chunk_idx"] = 0 # 下一章从0开始
        save_status(status)


    status["processed_files_map"][filepath] = "completed"
    # 当一本书的所有章节都完成后，可以清除 current_chapter_name_being_processed
    status["current_chapter_name_being_processed"] = None 
    save_status(status)
    print(f"  Finished processing book: {book_name}. Total triples from this book in this run: {current_book_triples_count}")
    return True


# --- 主函数 (核心修改) ---
def main():
    client = get_llm_client()
    status = load_status()
    
    csv_file_needs_header = [not os.path.exists(OUTPUT_CSV_FILE) or os.path.getsize(OUTPUT_CSV_FILE) == 0]

    output_dir = os.path.dirname(OUTPUT_CSV_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    book_id_counter = 0 # 用于给每本书一个简单的顺序ID
    processed_in_this_run_count = 0
    
    files_to_process_this_session = []
    # 修改文件遍历逻辑：不再遍历子文件夹
    if os.path.isdir(BASE_BOOKS_DIR):
        for filename in os.listdir(BASE_BOOKS_DIR):
            if filename.endswith(".txt"):
                files_to_process_this_session.append(os.path.join(BASE_BOOKS_DIR, filename))
    else:
        print(f"Error: Base directory '{BASE_BOOKS_DIR}' not found or is not a directory.")
        return

    files_to_process_this_session.sort()
    total_files = len(files_to_process_this_session)
    print(f"Found {total_files} .txt files to potentially process in '{BASE_BOOKS_DIR}'.")

    start_processing_from_file_idx = 0
    if status.get("current_file_path") and status["processed_files_map"].get(status["current_file_path"]) != "completed":
        try:
            start_processing_from_file_idx = files_to_process_this_session.index(status["current_file_path"])
            print(f"Attempting to resume from file: {status['current_file_path']} (index {start_processing_from_file_idx})")
        except ValueError:
            print(f"Warning: Status file's current_file_path '{status['current_file_path']}' not found in current scan. Resetting resume point.")
            status["current_file_path"] = None
            status["current_chapter_name_being_processed"] = None
            status["current_chunk_idx"] = 0
            # save_status(status) # 不在这里保存，避免覆盖有效状态，让它从头开始找未完成的
            start_processing_from_file_idx = 0 # 从列表头开始检查哪个未完成

    try:
        with open(OUTPUT_CSV_FILE, 'a+', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            if csv_file_needs_header[0] and csvfile.tell() == 0:
                writer.writerow(['Subject', 'Predicate', 'Object', 'SourceBookName', 'SourceChapterName'])
                csv_file_needs_header[0] = False

            for file_idx, filepath in enumerate(files_to_process_this_session):
                book_name_from_file = os.path.splitext(os.path.basename(filepath))[0]
                
                # 优先检查是否已完成
                if status["processed_files_map"].get(filepath) == "completed":
                    print(f"Skipping already completed file: {book_name_from_file} ({filepath})")
                    continue
                
                # 如果不是要恢复的文件，并且 resume point 存在且还没到，则跳过
                if status.get("current_file_path") and filepath != status.get("current_file_path") and file_idx < start_processing_from_file_idx :
                    print(f"Skipping file {book_name_from_file} as it's before the resume point file '{status.get('current_file_path')}'.")
                    continue
                
                # 如果状态中的 current_file_path 已完成，但由于某种原因没有被正确清除，
                # 而我们现在遍历到了一个新文件，应该重置状态以处理这个新文件。
                # 或者，如果 current_file_path 为 None (表示从头开始或上一本已完成)，则正常处理。
                if status.get("current_file_path") and status.get("current_file_path") != filepath and status["processed_files_map"].get(status["current_file_path"]) == "completed":
                    print(f"Previous resume file '{status.get('current_file_path')}' was completed. Moving to next file.")
                    status["current_file_path"] = filepath # 更新为当前文件
                    status["current_chapter_name_being_processed"] = None
                    status["current_chunk_idx"] = 0
                    # save_status(status) # process_book_file_v2 会在开始时保存


                book_id_counter += 1
                success = process_book_file_v2(filepath, book_name_from_file, book_id_counter, client, status, writer, csv_file_needs_header)
                if success:
                    processed_in_this_run_count += 1
                else:
                    print(f"Processing may have failed or encountered errors for book: {book_name_from_file}. Check logs. Status saved.")
            
            status["current_file_path"] = None
            status["current_chapter_name_being_processed"] = None
            status["current_chunk_idx"] = 0
            save_status(status)


    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current status...")
        save_status(status)
        print("Status saved. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
        save_status(status)
        print("Status saved due to error. Exiting.")
    finally:
        print(f"\n\nProcessing session finished.")
        print(f"Total books processed or attempted in this run: {processed_in_this_run_count}")
        print(f"All extracted triples are in '{OUTPUT_CSV_FILE}'.")
        print(f"Final processing status saved in '{STATUS_FILE}'.")

if __name__ == '__main__':
    main()

import os
import json
import time
import uuid
import zipfile
import shutil
import re
import asyncio
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from bson import ObjectId
import pymongo
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# ---------- 配置 ----------
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN")
MINERU_BASE_URL = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_DEFAULT_MODEL", "qwen-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))
MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "kb002")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)
QUESTIONS_COLLECTION = "quiz_questions"
QUIZZES_COLLECTION = "quizzes"
CHUNK_SIZE = 3000
OVERLAP = 100

app = FastAPI(title="多课程选择题刷题系统 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MongoDB 工具 ----------
def get_db():
    import ssl
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        client = pymongo.MongoClient(
            MONGO_URL,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        client.admin.command('ping')
        print("[MongoDB] 连接成功！")
        return client[MONGO_DB_NAME]
    except Exception as e:
        print(f"[MongoDB] 连接失败: {e}")
        raise e

def get_questions_collection():
    return get_db()[QUESTIONS_COLLECTION]

def get_quizzes_collection():
    return get_db()[QUIZZES_COLLECTION]

# ---------- 题库 CRUD ----------
def create_quiz(name: str) -> dict:
    coll = get_quizzes_collection()
    if coll.find_one({"name": name}):
        raise HTTPException(400, f"题库名称 '{name}' 已存在")
    doc = {
        "name": name,
        "created_at": datetime.utcnow()
    }
    result = coll.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    doc["created_at"] = doc["created_at"].isoformat()
    return doc

def list_quizzes() -> List[dict]:
    coll = get_quizzes_collection()
    quizzes = []
    for doc in coll.find().sort("created_at", -1):
        doc["_id"] = str(doc["_id"])
        doc["created_at"] = doc["created_at"].isoformat() if isinstance(doc["created_at"], datetime) else doc["created_at"]
        quizzes.append(doc)
    return quizzes

# ---------- 重叠分段工具 ----------
def split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks

# ---------- 原始同步函数（保留） ----------
def upload_pdf_to_mineru(file_bytes: bytes, filename: str) -> str:
    """原始同步上传，无进度回调"""
    if not MINERU_API_TOKEN:
        raise ValueError("MINERU_API_TOKEN 未设置")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }

    print("[MinerU] 正在获取上传链接...")
    batch_resp = requests.post(
        f"{MINERU_BASE_URL}/file-urls/batch",
        json={
            "files": [{"name": filename}],
            "model_version": "vlm"
        },
        headers=headers
    )
    batch_resp.raise_for_status()
    batch_data = batch_resp.json()
    if batch_data.get("code") != 0:
        raise RuntimeError(f"获取上传URL失败: {batch_data.get('msg', '未知错误')}")

    uploaded_url = batch_data["data"]["file_urls"][0]
    batch_id = batch_data["data"]["batch_id"]

    print("[MinerU] 正在上传 PDF...")
    session = requests.Session()
    session.trust_env = False
    try:
        put_resp = session.put(uploaded_url, data=file_bytes)
        put_resp.raise_for_status()
    finally:
        session.close()

    print("[MinerU] 任务处理中，请稍候...")
    result_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    timeout = 600
    start = time.time()
    zip_url = None
    while True:
        if time.time() - start > timeout:
            raise TimeoutError("MinerU 解析超时")
        time.sleep(3)
        res = requests.get(result_url, headers=headers)
        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                continue
            raise RuntimeError(f"查询状态失败，状态码: {res.status_code}")
        data = res.json()
        if data.get("code") != 0:
            raise RuntimeError(f"查询状态失败: {data.get('msg', '未知错误')}")
        extract_result = data["data"]["extract_result"][0]
        if extract_result["state"] == "done":
            zip_url = extract_result["full_zip_url"]
            print(f"[MinerU] 解析完成，用时 {time.time()-start:.1f}s")
            break

    print("[MinerU] 下载解析结果...")
    zip_resp = requests.get(zip_url)
    zip_resp.raise_for_status()

    stem = Path(filename).stem
    extract_dir = OUTPUT_DIR / stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / f"{stem}_result.zip"
    zip_path.write_bytes(zip_resp.content)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    md_files = list(extract_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError("解析结果中未找到 md 文件")

    target = None
    for f in md_files:
        if f.name == f"{stem}.md":
            target = f
            break
    if not target:
        for f in md_files:
            if f.name.lower() == "full.md":
                target = f
                break
    if not target:
        target = md_files[0]

    final_md_path = OUTPUT_DIR / f"{stem}.md"
    if target != final_md_path:
        target.rename(final_md_path)
    zip_path.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    markdown_content = final_md_path.read_text(encoding="utf-8")
    print(f"[MinerU] Markdown 已保存至 {final_md_path}")
    return markdown_content

def parse_questions_with_llm(markdown_text: str) -> List[Dict]:
    """原始 LLM 解析，无进度回调"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 未设置")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    system_prompt = (
        "你是一个专业的题目解析助手。从提供的文本中提取所有选择题（单选、多选）和判断题，为每道题生成简短解析。主观题和填空题不要理会。\n"
        "严格按以下 JSON 数组格式输出，不要包含任何其他文字：\n"
        "[\n"
        "  {\n"
        '    "type": "single_choice" | "multiple_choice" | "true_false",\n'
        '    "question": "完整的题目文字",\n'
        '    "options": {"A": "...", "B": "...", ...},\n'
        '    "answer": "正确选项字母，如 A; 多选题答案为多个字母连在一起，如 ABC; 判断题为 A 或 B",\n'
        '    "explanation": "解析，限200字以内"\n'
        "  },\n"
        "  ...\n"
        "]\n"
        "注意：\n"
        "- 只提取文本中完整出现的题目，不要创造不存在的题目。\n"
        "- 单选题只有一个正确选项，多选题有多个正确选项。\n"
        "- 判断题的选项固定为 {\"A\":\"正确\", \"B\":\"错误\"}，答案必须是 A 或 B。\n"
        "- 解析必须详细，但不超过200个汉字。\n"
        "如果文本中不含选择题/判断题，返回 [] 。"
    )

    chunks = split_text_with_overlap(markdown_text, CHUNK_SIZE, OVERLAP)
    all_questions = []
    seen_questions = set()

    for idx, chunk in enumerate(chunks):
        print(f"[LLM] 正在解析第 {idx+1}/{len(chunks)} 段...")
        user_prompt = f"请从以下Markdown内容（第{idx+1}段）提取所有选择题/判断题：\n\n{chunk}"
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            try:
                questions = json.loads(content)
            except json.JSONDecodeError:
                print(f"第{idx+1}段 JSON 解析失败，尝试修复...")
                repaired = repair_json_string(content)
                questions = json.loads(repaired)
                print("修复后解析成功")

            if not isinstance(questions, list):
                print(f"警告：第{idx+1}段返回的不是数组，已跳过")
                continue
            for q in questions:
                q_text = q.get("question", "")
                if q_text and q_text not in seen_questions:
                    seen_questions.add(q_text)
                    all_questions.append(q)
            print(f"第{idx+1}段解析出 {len(questions)} 题，累计有效 {len(all_questions)} 题")
        except json.JSONDecodeError as e:
            print(f"第{idx+1}段 JSON 解析失败（修复后仍失败）：{e}，跳过该段")
            continue
        except Exception as e:
            print(f"第{idx+1}段解析异常：{e}，跳过该段")
            continue

    print(f"[LLM] 全部解析完成，共提取 {len(all_questions)} 道题目")
    return all_questions

def repair_json_string(json_str: str) -> str:
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    json_str = re.sub(r'"\s*\n\s*{', '",\n{', json_str)
    json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
    return json_str

def save_questions_to_db(questions: List[Dict], quiz_id: str):
    if not questions:
        return 0
    coll = get_questions_collection()
    inserted = 0
    for q in questions:
        if not coll.find_one({"quiz_id": quiz_id, "question": q["question"]}):
            q["quiz_id"] = quiz_id
            coll.insert_one(q)
            inserted += 1
    return inserted

# ---------- 带进度回调的包装函数 ----------
def upload_pdf_to_mineru_with_progress(file_bytes: bytes, filename: str, progress_callback):
    """在同步线程中执行，并通过 callback 报告进度"""
    if not MINERU_API_TOKEN:
        raise ValueError("MINERU_API_TOKEN 未设置")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }

    progress_callback({"msg": "正在获取上传链接...", "progress": 5})
    batch_resp = requests.post(
        f"{MINERU_BASE_URL}/file-urls/batch",
        json={
            "files": [{"name": filename}],
            "model_version": "vlm"
        },
        headers=headers
    )
    batch_resp.raise_for_status()
    batch_data = batch_resp.json()
    if batch_data.get("code") != 0:
        raise RuntimeError(f"获取上传URL失败: {batch_data.get('msg', '未知错误')}")

    uploaded_url = batch_data["data"]["file_urls"][0]
    batch_id = batch_data["data"]["batch_id"]

    progress_callback({"msg": "正在上传 PDF...", "progress": 15})
    session = requests.Session()
    session.trust_env = False
    try:
        put_resp = session.put(uploaded_url, data=file_bytes)
        put_resp.raise_for_status()
    finally:
        session.close()

    progress_callback({"msg": "上传完成，等待 MinerU 解析...", "progress": 25})
    result_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    timeout = 600
    start = time.time()
    zip_url = None
    base_progress = 25
    max_wait_progress = 70
    elapsed = 0
    while True:
        if time.time() - start > timeout:
            raise TimeoutError("MinerU 解析超时")
        time.sleep(3)
        elapsed += 3
        p = min(max_wait_progress, base_progress + int(elapsed / timeout * (max_wait_progress - base_progress)))
        progress_callback({"msg": f"MinerU 解析中 ({elapsed}s)", "progress": p})

        res = requests.get(result_url, headers=headers)
        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                continue
            raise RuntimeError(f"查询状态失败，状态码: {res.status_code}")
        data = res.json()
        if data.get("code") != 0:
            raise RuntimeError(f"查询状态失败: {data.get('msg', '未知错误')}")
        extract_result = data["data"]["extract_result"][0]
        if extract_result["state"] == "done":
            zip_url = extract_result["full_zip_url"]
            progress_callback({"msg": f"MinerU 解析完成，用时 {elapsed}s", "progress": 75})
            break

    progress_callback({"msg": "下载解析结果...", "progress": 78})
    zip_resp = requests.get(zip_url)
    zip_resp.raise_for_status()

    stem = Path(filename).stem
    extract_dir = OUTPUT_DIR / stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / f"{stem}_result.zip"
    zip_path.write_bytes(zip_resp.content)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    md_files = list(extract_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError("解析结果中未找到 md 文件")

    target = None
    for f in md_files:
        if f.name == f"{stem}.md":
            target = f
            break
    if not target:
        for f in md_files:
            if f.name.lower() == "full.md":
                target = f
                break
    if not target:
        target = md_files[0]

    final_md_path = OUTPUT_DIR / f"{stem}.md"
    if target != final_md_path:
        target.rename(final_md_path)
    zip_path.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    markdown_content = final_md_path.read_text(encoding="utf-8")
    progress_callback({"msg": "Markdown 提取完成", "progress": 85})
    return markdown_content

def parse_questions_with_llm_with_progress(markdown_text: str, progress_callback):
    """带进度的 LLM 解析"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 未设置")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    system_prompt = (
        "你是一个专业的题目解析助手。从提供的文本中提取所有选择题（单选、多选）和判断题，为每道题生成简短解析。主观题和填空题不要理会。\n"
        "严格按以下 JSON 数组格式输出，不要包含任何其他文字：\n"
        "[\n"
        "  {\n"
        '    "type": "single_choice" | "multiple_choice" | "true_false",\n'
        '    "question": "完整的题目文字",\n'
        '    "options": {"A": "...", "B": "...", ...},\n'
        '    "answer": "正确选项字母，如 A; 多选题答案为多个字母连在一起，如 ABC; 判断题为 A 或 B",\n'
        '    "explanation": "解析，限200字以内"\n'
        "  },\n"
        "  ...\n"
        "]\n"
        "注意：\n"
        "- 只提取文本中完整出现的题目，不要创造不存在的题目。\n"
        "- 单选题只有一个正确选项，多选题有多个正确选项。\n"
        "- 判断题的选项固定为 {\"A\":\"正确\", \"B\":\"错误\"}，答案必须是 A 或 B。\n"
        "- 解析必须详细，但不超过200个汉字。\n"
        "如果文本中不含选择题/判断题，返回 [] 。"
    )

    chunks = split_text_with_overlap(markdown_text, CHUNK_SIZE, OVERLAP)
    all_questions = []
    seen_questions = set()

    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks):
        p = 85 + int((idx / total_chunks) * 10)  # 85% ~ 95%
        progress_callback({"msg": f"AI 提取题目中 ({idx+1}/{total_chunks})", "progress": p})
        user_prompt = f"请从以下Markdown内容（第{idx+1}段）提取所有选择题/判断题：\n\n{chunk}"
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            try:
                questions = json.loads(content)
            except json.JSONDecodeError:
                repaired = repair_json_string(content)
                questions = json.loads(repaired)

            if not isinstance(questions, list):
                continue
            for q in questions:
                q_text = q.get("question", "")
                if q_text and q_text not in seen_questions:
                    seen_questions.add(q_text)
                    all_questions.append(q)
        except Exception as e:
            print(f"第{idx+1}段解析异常：{e}，跳过")

    progress_callback({"msg": f"AI 提取完成，共 {len(all_questions)} 题", "progress": 95})
    return all_questions

# ---------- 流式路由（修复版） ----------
@app.post("/api/import-pdf-stream")
async def import_pdf_stream(file: UploadFile = File(...), quiz_id: str = Form(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "只支持 PDF 文件")

    if not get_quizzes_collection().find_one({"_id": ObjectId(quiz_id)}):
        raise HTTPException(400, "所选题库不存在")

    async def event_generator():
        queue = asyncio.Queue(maxsize=100)
        worker_done = asyncio.Future()
        loop = asyncio.get_event_loop()

        def progress_callback(msg_dict):
            try:
                queue.put_nowait(msg_dict)
            except asyncio.QueueFull:
                pass

        def do_work():
            try:
                file_bytes = file.file.read()
                progress_callback({"msg": "文件读取完成", "progress": 2})

                md_content = upload_pdf_to_mineru_with_progress(file_bytes, file.filename, progress_callback)

                # 🧹 删除临时生成的 .md 文件（内容已保存在 md_content 变量中）
                stem = Path(file.filename).stem
                md_file = OUTPUT_DIR / f"{stem}.md"
                if md_file.exists():
                    md_file.unlink()
                    print(f"[清理] 已删除临时文件: {md_file}")

                questions = parse_questions_with_llm_with_progress(md_content, progress_callback)

                inserted = save_questions_to_db(questions, quiz_id)
                final_msg = f"成功导入 {inserted} 道新题目（共解析 {len(questions)} 道）"
                progress_callback({"msg": final_msg, "progress": 100, "done": True})
            except Exception as e:
                progress_callback({"msg": f"错误: {str(e)}", "progress": -1, "error": True})
            finally:
                loop.call_soon_threadsafe(worker_done.set_result, None)

        await loop.run_in_executor(None, do_work)

        while True:
            if worker_done.done() and queue.empty():
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item.get("done"):
                yield f"data: {json.dumps({'step': 'done', 'message': item['msg'], 'progress': item['progress']}, ensure_ascii=False)}\n\n"
                break
            if item.get("error"):
                yield f"data: {json.dumps({'step': 'error', 'message': item['msg'], 'progress': -1}, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps({'step': 'progress', 'message': item['msg'], 'progress': item['progress']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/import-md-stream")
async def import_md_stream(request: Request):
    data = await request.json()
    md_content = data.get("content")
    filename = data.get("filename", "unknown.md")
    quiz_id = data.get("quiz_id")

    if not md_content or not quiz_id:
        raise HTTPException(400, "缺少 content 或 quiz_id")

    if not get_quizzes_collection().find_one({"_id": ObjectId(quiz_id)}):
        raise HTTPException(400, "所选题库不存在")

    async def event_generator():
        queue = asyncio.Queue(maxsize=100)
        worker_done = asyncio.Future()
        loop = asyncio.get_event_loop()

        def progress_callback(msg_dict):
            try:
                queue.put_nowait(msg_dict)
            except asyncio.QueueFull:
                pass

        def do_work():
            try:
                progress_callback({"msg": "文件读取完成", "progress": 5})
                # 直接调用 LLM 解析（无 MinerU）
                questions = parse_questions_with_llm_with_progress(md_content, progress_callback)
                inserted = save_questions_to_db(questions, quiz_id)
                final_msg = f"成功导入 {inserted} 道新题目（共解析 {len(questions)} 道）"
                progress_callback({"msg": final_msg, "progress": 100, "done": True})
            except Exception as e:
                progress_callback({"msg": f"错误: {str(e)}", "progress": -1, "error": True})
            finally:
                loop.call_soon_threadsafe(worker_done.set_result, None)

        await loop.run_in_executor(None, do_work)

        while True:
            if worker_done.done() and queue.empty():
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item.get("done"):
                yield f"data: {json.dumps({'step': 'done', 'message': item['msg'], 'progress': item['progress']}, ensure_ascii=False)}\n\n"
                break
            if item.get("error"):
                yield f"data: {json.dumps({'step': 'error', 'message': item['msg'], 'progress': -1}, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps({'step': 'progress', 'message': item['msg'], 'progress': item['progress']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------- 原有同步接口（保留） ----------
@app.post("/api/quizzes")
async def api_create_quiz(name: str = Form(...)):
    try:
        quiz = await asyncio.to_thread(create_quiz, name)
        return JSONResponse({"success": True, "quiz": quiz})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/quizzes")
async def api_list_quizzes():
    quizzes = await asyncio.to_thread(list_quizzes)
    return JSONResponse(quizzes)

@app.post("/api/import-pdf")  # 保留原同步接口
async def import_pdf(file: UploadFile = File(...), quiz_id: str = Form(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "只支持 PDF 文件")
    if not get_quizzes_collection().find_one({"_id": ObjectId(quiz_id)}):
        raise HTTPException(400, "所选题库不存在")
    try:
        file_bytes = await file.read()
        md_content = await asyncio.to_thread(upload_pdf_to_mineru, file_bytes, file.filename)
        questions = await asyncio.to_thread(parse_questions_with_llm, md_content)
        inserted = await asyncio.to_thread(save_questions_to_db, questions, quiz_id)
        return JSONResponse({
            "success": True,
            "message": f"成功导入 {inserted} 道新题目（共解析 {len(questions)} 道）"
        })
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/api/db-test")
async def test_db():
    try:
        db = get_db()
        result = db.list_collection_names()
        return {"status": "connected", "collections": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/start-quiz")
async def start_quiz(quiz_id: str, count: int = 30):
    if not get_quizzes_collection().find_one({"_id": ObjectId(quiz_id)}):
        raise HTTPException(400, "题库不存在")
    if count < 1:
        raise HTTPException(400, "题目数量必须大于0")
    coll = get_questions_collection()
    total = coll.count_documents({"quiz_id": quiz_id})
    if total == 0:
        raise HTTPException(404, "该题库为空，请先导入题目")
    sample_size = min(count, total)
    pipeline = [
        {"$match": {"quiz_id": quiz_id}},
        {"$sample": {"size": sample_size}}
    ]
    docs = list(coll.aggregate(pipeline))
    if not docs:
        raise HTTPException(404, "该题库为空，请先导入题目")
    session_id = str(uuid.uuid4())
    answer_cache[session_id] = {}
    questions_response = []
    for doc in docs:
        qid = str(doc["_id"])
        answer_cache[session_id][qid] = {
            "type": doc.get("type", "single_choice"),
            "answer": doc["answer"],
            "explanation": doc.get("explanation", "")
        }
        questions_response.append({
            "id": qid,
            "type": doc.get("type", "single_choice"),
            "question": doc["question"],
            "options": doc["options"]
        })
    return JSONResponse({"quiz_id": session_id, "questions": questions_response})

@app.get("/api/start-sequential-quiz")
async def start_sequential_quiz(quiz_id: str):
    if not get_quizzes_collection().find_one({"_id": ObjectId(quiz_id)}):
        raise HTTPException(400, "题库不存在")
    coll = get_questions_collection()
    docs = list(coll.find({"quiz_id": quiz_id}).sort("_id", 1))
    if not docs:
        raise HTTPException(404, "该题库为空，请先导入题目")
    questions_response = []
    for doc in docs:
        qid = str(doc["_id"])
        questions_response.append({
            "id": qid,
            "type": doc.get("type", "single_choice"),
            "question": doc["question"],
            "options": doc["options"]
        })
    return JSONResponse({"questions": questions_response})

class SingleAnswerRequest(BaseModel):
    question_id: str
    answer: str

@app.post("/api/check-single-answer")
async def check_single_answer(req: SingleAnswerRequest):
    coll = get_questions_collection()
    try:
        doc = coll.find_one({"_id": ObjectId(req.question_id)})
    except:
        raise HTTPException(400, "无效的题目ID")
    if not doc:
        raise HTTPException(404, "题目不存在")
    correct = doc["answer"]
    explanation = doc.get("explanation", "")
    qtype = doc.get("type", "single_choice")
    user = req.answer.strip().upper()
    cor = correct.strip().upper()
    if qtype in ("single_choice", "true_false"):
        is_correct = (user == cor)
    elif qtype == "multiple_choice":
        is_correct = ''.join(sorted(user)) == ''.join(sorted(cor))
    else:
        is_correct = (user == cor)
    return JSONResponse({
        "is_correct": is_correct,
        "correct_answer": correct,
        "explanation": explanation
    })

class AnswerItem(BaseModel):
    id: str
    answer: str

class QuizSubmitRequest(BaseModel):
    quiz_id: str
    answers: List[AnswerItem]

def is_answer_correct(correct: str, user: str, qtype: str) -> bool:
    u = user.strip().upper()
    c = correct.strip().upper()
    if qtype in ("single_choice", "true_false"):
        return u == c
    elif qtype == "multiple_choice":
        return ''.join(sorted(u)) == ''.join(sorted(c))
    else:
        return u == c

answer_cache: Dict[str, Dict] = {}

@app.post("/api/submit-quiz")
async def submit_quiz(req: QuizSubmitRequest):
    cached = answer_cache.pop(req.quiz_id, None)
    if cached is None:
        raise HTTPException(400, "答题会话已过期或不存在")
    results = []
    correct_count = 0
    for item in req.answers:
        info = cached.get(item.id)
        if not info:
            results.append({"id": item.id, "user_answer": item.answer, "correct_answer": "?", "explanation": "题目不存在", "is_correct": False})
            continue
        qtype = info.get("type", "single_choice")
        correct = info["answer"]
        user_ans = item.answer
        is_correct = is_answer_correct(correct, user_ans, qtype)
        if is_correct:
            correct_count += 1
        results.append({
            "id": item.id, "user_answer": user_ans, "correct_answer": correct,
            "explanation": info["explanation"], "is_correct": is_correct
        })
    return JSONResponse({
        "total": len(req.answers),
        "correct": correct_count,
        "results": results
    })

@app.get("/")
async def root():
    return {"message": "多课程选择题刷题系统 API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

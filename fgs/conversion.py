"""Loss-aware JSON/JSONL to UTF-8 CSV conversion, shared by CLI and local UI."""
import csv
import io
import json
import math
from pathlib import Path


def parse_records(text, records_key=None, chat_pairs=False):
    text = text.lstrip("\ufeff")
    if not text.strip():
        raise ValueError("輸入檔案沒有資料")
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f"JSON 物件有重複欄位：{k}")
            result[k] = v
        return result
    def invalid(value):
        raise ValueError(f"JSON 不允許非有限數值：{value}")
    def decode(value):
        return json.loads(value, object_pairs_hook=unique, parse_constant=invalid)
    try:
        value = decode(text)
    except json.JSONDecodeError:
        try:
            value = [decode(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError("不是有效的 JSON 或 JSONL") from exc
    if records_key:
        if not isinstance(value, dict) or records_key not in value:
            raise ValueError("找不到指定的資料陣列欄位")
        value = value[records_key]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value or any(not isinstance(r, dict) for r in value):
        raise ValueError("資料必須是物件、物件陣列，或每行一個物件的 JSONL")
    def finite(x):
        if isinstance(x, float) and not math.isfinite(x):
            invalid(str(x))
        if isinstance(x, dict):
            for v in x.values():
                finite(v)
        if isinstance(x, list):
            for v in x:
                finite(v)
    finite(value)
    if chat_pairs:
        converted = []
        for row in value:
            messages = row.get("messages")
            if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
                raise ValueError("問答模式需要 messages 陣列")
            turns = [m for m in messages if m.get("role") in {"user", "assistant"}]
            if len(turns) != 2 or [m["role"] for m in turns] != ["user", "assistant"] or any(not isinstance(m.get("content"), str) for m in turns):
                raise ValueError("問答模式只接受一個 user 接一個 assistant；多輪對話請使用一般模式保留內容")
            if "question" in row or "answer" in row:
                raise ValueError("已有 question/answer 欄位，不能覆寫")
            converted.append(dict(row, question=turns[0]["content"], answer=turns[1]["content"]))
        value = converted
    return value


def csv_text(rows, columns=None, excel_safe=True):
    fields = columns or list(dict.fromkeys(k for row in rows for k in row))
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("CSV 欄位必須非空且不重複")
    if columns and set(columns) - {k for row in rows for k in row}:
        raise ValueError("指定欄位不存在")
    def cell(value):
        if value is None:
            return ""
        if isinstance(value, (dict, list, bool)):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, str) and excel_safe and (value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r"))):
            return "'" + value
        return value
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow([cell(f) for f in fields])
    writer.writerows([[cell(row.get(k)) for k in fields] for row in rows])
    return stream.getvalue(), fields


def convert_json(input_path, out, records_key=None, columns=None, chat_pairs=False, excel_safe=True):
    source, target = Path(input_path), Path(out)
    rows = parse_records(source.read_text(encoding="utf-8-sig"), records_key, chat_pairs)
    text, fields = csv_text(rows, columns, excel_safe)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8-sig", newline="") as handle:
        handle.write(text)
    return {"output": str(target.resolve()), "rows": len(rows), "columns": fields, "excel_safe": excel_safe,
            "note": "Nested objects/arrays remain JSON text. null/missing become empty cells. Formula-like strings gain an apostrophe in Excel-safe mode."}

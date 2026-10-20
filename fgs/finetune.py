"""Auditable supervised fine-tuning submission/status/download. Never auto-retry jobs."""
import json
import os
from pathlib import Path
import urllib.request
from urllib.parse import urlparse, quote
import uuid

from .io import digest, file_hash, new_output, read_json, read_jsonl, write_json


class FineTuneClient:
    def __init__(self, config, live=False):
        self.config, self.live = config, live
        self.base = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        if urlparse(self.base).scheme != "https":
            raise ValueError("Fine-tuning endpoint requires HTTPS")
        if "api_key" in config:
            raise ValueError("Use api_key_env, not plaintext secrets")

    def request(self, method, path, payload=None, raw=None, content_type=None, binary=False):
        if not self.live:
            raise ValueError("Fine-tuning network operations require --live")
        key = os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY"))
        if not key:
            raise ValueError("Missing API key environment variable")
        data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": content_type or "application/json"})
        with urllib.request.urlopen(request, timeout=float(self.config.get("timeout_seconds", 120))) as response:
            body = response.read()
        return body if binary else json.loads(body)

    def upload(self, path):
        boundary = uuid.uuid4().hex
        raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nfine-tune\r\n'
               f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="dataset.jsonl"\r\n'
               'Content-Type: application/jsonl\r\n\r\n').encode() + Path(path).read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        return self.request("POST", "/files", raw=raw, content_type=f"multipart/form-data; boundary={boundary}")


def prepare_job(train, validation, config_path, out):
    config = read_json(config_path) if not isinstance(config_path, dict) else dict(config_path)
    allowed = {"base_url", "api_key_env", "base_model", "seed", "hyperparameters", "task", "timeout_seconds"}
    if set(config) - allowed:
        raise ValueError("Unexpected fine-tune config field")
    if not config.get("base_model") or config["base_model"].startswith("ft:"):
        raise ValueError("Restart training from an explicit base model, not an old fine-tuned model")
    if config.get("task") not in {"generation", "refinement"}:
        raise ValueError("task must be generation or refinement")
    hp = config.get("hyperparameters", {})
    if set(hp) - {"n_epochs", "batch_size", "learning_rate_multiplier"}:
        raise ValueError("Unsupported supervised hyperparameter")
    for key, value in hp.items():
        if value != "auto" and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"Invalid hyperparameter {key}")
    user_sets = []
    for path in [train, validation]:
        rows = read_jsonl(path)
        if not rows:
            raise ValueError("Training and validation files must both contain records")
        texts = set()
        for row in rows:
            ms = row.get("messages", [])
            if not ms or ms[-1].get("role") != "assistant" or not any(m.get("role") == "user" for m in ms):
                raise ValueError("Invalid fine-tune chat record")
            texts.update(m["content"].strip() for m in ms if m["role"] in {"user", "assistant"})
        user_sets.append(texts)
    if user_sets[0] & user_sets[1]:
        raise ValueError("Train/validation chat content overlap; use the joint dataset split")
    out = new_output(out)
    for name, path in [("train", train), ("validation", validation)]:
        (out / f"{name}.jsonl").write_bytes(Path(path).read_bytes())
    inputs = {"config": config, "train_sha256": file_hash(train), "validation_sha256": file_hash(validation)}
    ledger = {**inputs, "fingerprint": digest(json.dumps(inputs, sort_keys=True)), "status": "prepared", "uploaded_files": {}}
    write_json(out / "job.json", ledger)
    return ledger


def submit_job(folder, live=False, client=None):
    folder = Path(folder)
    ledger = read_json(folder / "job.json")
    for split in ["train", "validation"]:
        if file_hash(folder / f"{split}.jsonl") != ledger[f"{split}_sha256"]:
            raise ValueError("Prepared fine-tune input changed")
    if ledger.get("job_id"):
        return ledger
    if ledger["status"] != "prepared":
        raise ValueError("Submission already attempted; use ft-sync to reconcile before any new job")
    client = client or FineTuneClient(ledger["config"], live)
    if not live:
        raise ValueError("Submitting training jobs requires --live")
    if ledger["config"]["base_model"].startswith("YOUR_"):
        raise ValueError("Configure a supported base model before uploading")
    ledger["status"] = "submitting"
    write_json(folder / "job.json", ledger)
    # Persist before each irreversible step; a network failure never silently duplicates a job.
    for split in ["train", "validation"]:
        if split not in ledger["uploaded_files"]:
            upload = client.upload(folder / f"{split}.jsonl")
            ledger["uploaded_files"][split] = upload["id"]
            write_json(folder / "job.json", ledger)
    cfg = ledger["config"]
    payload = {"model": cfg["base_model"], "training_file": ledger["uploaded_files"]["train"],
               "validation_file": ledger["uploaded_files"]["validation"], "seed": cfg.get("seed", 42),
               "method": {"type": "supervised", "supervised": {"hyperparameters": cfg.get("hyperparameters", {})}},
               "metadata": {"fgs_fingerprint": ledger["fingerprint"], "fgs_task": cfg["task"]}}
    job = client.request("POST", "/fine_tuning/jobs", payload)
    ledger.update(job_id=job["id"], status=job["status"], provider_job=job)
    write_json(folder / "job.json", ledger)
    return ledger


def sync_job(folder, live=False, client=None):
    folder = Path(folder)
    ledger = read_json(folder / "job.json")
    client = client or FineTuneClient(ledger["config"], live)
    if not live:
        raise ValueError("Retrieving fine-tuning status requires --live")
    if not ledger.get("job_id"):
        matches, after = [], ""
        for _ in range(20):
            page = client.request("GET", "/fine_tuning/jobs?limit=100" + ("&after=" + quote(after) if after else ""))
            matches.extend(j for j in page["data"] if (j.get("metadata") or {}).get("fgs_fingerprint") == ledger["fingerprint"])
            if not page.get("has_more") or not page["data"]:
                break
            after = page["data"][-1]["id"]
        if len(matches) != 1:
            raise ValueError("Could not uniquely reconcile submission. Inspect provider jobs; do not automatically resubmit")
        ledger["job_id"] = matches[0]["id"]
    job = client.request("GET", "/fine_tuning/jobs/" + quote(ledger["job_id"], safe=""))
    ledger.update(status=job["status"], provider_job=job, fine_tuned_model=job.get("fine_tuned_model"))
    write_json(folder / "job.json", ledger)
    if job["status"] == "succeeded":
        for i, file_id in enumerate(job.get("result_files", [])):
            path = folder / f"training_results_{i}.csv"
            if not path.exists():
                path.write_bytes(client.request("GET", "/files/" + quote(file_id, safe="") + "/content", binary=True))
    return ledger

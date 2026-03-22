# Burmese Hybrid ELIZA Chat UI Guide

This project includes a local browser-based chat interface for a Burmese Hybrid ELIZA chatbot. The UI serves a simple web app from Python and connects to a rule-based ELIZA engine with optional LSTM-based emotion detection.

If the model checkpoint is available, the chatbot can load the Burmese emotion model. If not, it still works in rule-based mode.

## Main File

All paths below are relative to the `group-2/` project root.

- `experiments/burmese_chat_ui.py` — local web server and chat UI
- `experiments/hybrid-eliza-improved-v1.0.py` — primary hybrid module (chatbot logic and LSTM loading); falls back to `experiments/hybrid-eliza-mm-bilstm-attention.py` if the primary file is missing
- `experiments/eliza_eq_mm_improve_v1.0.pth` — optional Burmese LSTM checkpoint (not tracked in git); place it next to the scripts or pass `--model_path`

## Requirements

- Python 3.11 or newer
- Project dependencies from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Run from the `group-2/` project root:

```bash
python experiments/burmese_chat_ui.py --model_path experiments/eliza_eq_mm_improve_v1.0.pth
```

Then open this in your browser:

```text
http://127.0.0.1:8765
```

## Run Without Model

If you want to use the chatbot without loading the LSTM checkpoint:

```bash
python experiments/burmese_chat_ui.py
```

In that case, the UI still works, but it falls back to rule-based replies only.

## Optional Arguments

```bash
python experiments/burmese_chat_ui.py --host 127.0.0.1 --port 8765 --lang mm --model_path experiments/eliza_eq_mm_improve_v1.0.pth
```

Available options:

- `--host` - server host, default is `127.0.0.1`
- `--port` - server port, default is `8765`
- `--lang` - `mm` or `en`
- `--model_path` - path to the `.pth` checkpoint

## Example

```bash
python experiments/burmese_chat_ui.py --host 0.0.0.0 --port 9000 --lang mm --model_path experiments/eliza_eq_mm_improve_v1.0.pth
```

Then open:

```text
http://127.0.0.1:9000
```


## Notes

- The UI defaults to `experiments/eliza_eq_mm_improve_v1.0.pth` for `--lang mm` when `--model_path` is omitted; pass `--model_path` if your checkpoint lives elsewhere.
- Press `Enter` to send a message in the UI.
- Press `Shift + Enter` to add a new line.

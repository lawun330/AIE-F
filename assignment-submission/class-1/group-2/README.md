# Group 2 Assignment Submission

## Dataset

Our custom dataset comprises Facebook comments, synthetic data generated via generative models, and manually authored entries. It is categorized into the following six emotional classes:
- 0: Sadness
- 1: Joy
- 2: Love
- 3: Anger
- 4: Fear
- 5: Surprise

## Project Structure

```
group-2/
├── group2-hybrid-eliza.py          # single CLI entry: --mode train|eval|chat
├── data/
│   ├── raw_ungrouped/              # original team contributions (not merged)
│   ├── annotated_ungrouped/        # cleaned team contributions (not merged)
│   ├── merged/                     # merged spreadsheets before final export
│   ├── merged_preprocessed/        # final merged CSVs used for training
│   │   ├── data_before_downsampling.csv
│   │   └── data_after_downsampling.csv
│   └── stopwords.txt               # Burmese stopword list (see Sources)
├── checkpoints/                    # saved checkpoints and tokenizer/vocab artifacts
├── notebooks/                      # experimental analysis and EDA
├── scripts/
│   ├── train.py                    # model training execution
│   ├── eval.py                     # model evaluation/prediction
│   └── chat.py                     # interactive inference; reuses eval helpers
├── src/
│   ├── preprocessing.py            # Burmese text normalization, tokenization (see Sources), stopword removal
│   ├── rabbit.py                   # Zawgyi to Unicode conversion utilities (see Sources)
│   ├── vocab_builder.py            # vocab/token-id and label-id helpers
│   ├── prep_data.py                # shared preprocessing helpers for train/eval/chat
│   └── model.py                    # LSTM architecture and layer definitions
├── README.md                       # documentation
├── environment.yaml                # conda environment configuration
└── requirements.txt                # dependencies
```

## Project Flow

The code is organized so a single wrapper script controls the high-level mode, while `src/` contains reusable building blocks.

- `group2-hybrid-eliza.py` (primary entry point): the top-level CLI wrapper/dispatcher. It selects one of the modes and starts the corresponding script in `scripts/`.

- `scripts/train.py`: training orchestration. It uses `src/prep_data.py` to build train/validation tensors and class weights, uses `src/model.py` to define the model, and saves a checkpoint that includes model weights plus vocabulary/label mappings.

- `scripts/eval.py`: evaluation/prediction on a labeled dataset. It loads the saved checkpoint, uses `src/prep_data.py` to convert raw texts into padded token-id tensors using the same preprocessing pipeline, then runs `src/model.py` to produce predictions and confidence scores.

- `scripts/chat.py`: interactive inference. It loads the saved checkpoint and reuses the same preprocessing + model inference path as `scripts/eval.py` for user-entered text, then prints the predicted emotion and confidence.

- `src/prep_data.py`: shared data preparation pipeline used by both training and inference. It connects dataset reading, text preprocessing, token/id encoding, padding/truncation, train/val splitting (for training), and class-weight computation; it relies on `src/preprocessing.py` (TextProcessor) and `src/vocab_builder.py` (vocabulary + label mapping utilities).

- `src/preprocessing.py`: text preprocessing pipeline. It performs Zawgyi-to-Unicode normalization when needed, regex punctuation cleanup, tokenization with the MMDT tokenizer, and stopword removal; it relies on `src/rabbit.py` for Zawgyi conversion and on the Myanmar detection/tokenization libraries.

- `src/vocab_builder.py`: vocabulary and label mapping utilities. It builds the `word -> id` vocabulary from training tokenized text and defines the fixed `label -> id` order for the six emotion classes.

- `src/model.py`: model definition. It embeds token ids, runs a bidirectional LSTM, then pools sequence information (either attention pooling or final-state pooling) and applies a linear classifier to get logits.

- `src/rabbit.py`: Zawgyi-to-Unicode conversion rules used by the preprocessing pipeline.

In short: `group2-hybrid-eliza.py` starts the run; `scripts/*.py` control train/eval/chat; `src/*.py` implements the reusable preprocessing/model utilities.

## Sources:
- Burmese grammar: https://online.fliphtml5.com/rrlzh/mbir/#p=1
- Rabbit Zawgyi to Unicode Converter: https://github.com/Rabbit-Converter/Rabbit-Python 
- MMDT Tokenizer: https://github.com/Myanmar-Data-Tech/mmdt-tokenizer
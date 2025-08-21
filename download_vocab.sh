#!/bin/bash
set -ex

# vocabファイルがなければダウンロード（失敗しても起動は継続）
python -c "from openai_harmony import load_harmony_encoding, HarmonyEncodingName; load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)" || true

# キャッシュディレクトリの中身を表示
ls -l /root/.cache/openai-harmony || true

exec "$@"


# syntax=docker/dockerfile:1
FROM python:3.12-slim

# システム依存パッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# キャッシュディレクトリ作成と権限付与
RUN mkdir -p /root/.cache/openai-harmony && chmod -R 777 /root/.cache
ENV OPENAI_HARMONY_CACHE_DIR=/root/.cache/openai-harmony

# 作業ディレクトリの作成
WORKDIR /app

# プロジェクトファイルのコピー
COPY . /app

# Python依存パッケージのインストール（公式ガイド＋transformers明示的追加＋openai-harmony最新版）
RUN pip install --upgrade pip setuptools requests certifi && \
    pip install openai-harmony --upgrade && \
    pip install .[torch] && \
    pip install transformers && \
    apt-get update && apt-get install -y curl && \
    curl -fSL -o /root/.cache/openai-harmony/harmony-gpt-oss.tiktoken https://openaipublic.blob.core.windows.net/harmony/v1/harmony-gpt-oss.tiktoken

# vocabダウンロードスクリプトをコピー
COPY download_vocab.sh /download_vocab.sh
RUN chmod +x /download_vocab.sh

# ポート番号（必要に応じて変更）
EXPOSE 8000

# サーバー起動例（transformersバックエンド指定）
ENTRYPOINT ["/download_vocab.sh"]
CMD ["python", "-m", "gpt_oss.responses_api.serve", "--inference-backend", "transformers"]

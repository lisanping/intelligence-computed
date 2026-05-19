#!/usr/bin/env bash
# ==============================================================================
# 第 16 讲 · 动手实验：用 Ollama 部署本地量化模型
# ==============================================================================
# 依赖：Ollama CLI (>= 0.3)  https://ollama.com/download
# 硬件：7B Q4 需要 ~8 GB RAM；70B Q4 需要 ~48 GB RAM
# 所有推理在本地完成，无需网络（模型下载除外）
# ==============================================================================

set -e

echo "=============================================="
echo " 第 22 讲 · Ollama 本地部署 Demo"
echo "=============================================="

# --------------------------------------------------
# 0. 检查 Ollama 是否安装
# --------------------------------------------------
if ! command -v ollama &> /dev/null; then
    echo "[ERROR] Ollama 未安装。请先安装："
    echo "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Windows:     https://ollama.com/download"
    exit 1
fi

echo "[OK] Ollama 版本: $(ollama --version)"
echo ""

# --------------------------------------------------
# 1. 拉取小模型（Qwen2.5-7B Q4）快速验证
# --------------------------------------------------
SMALL_MODEL="qwen2.5:7b-instruct-q4_K_M"
echo ">>> 步骤 1: 拉取小模型 ${SMALL_MODEL} (~4.4 GB) ..."
ollama pull "${SMALL_MODEL}"
echo "[OK] 小模型已就绪。"
echo ""

# --------------------------------------------------
# 2. 基础对话测试
# --------------------------------------------------
echo ">>> 步骤 2: 基础对话测试 ..."
echo ""

echo "--- 测试 1: 解释量化 ---"
ollama run "${SMALL_MODEL}" "用一句话解释什么是深度学习中的量化（quantization）。" <<< ""
echo ""

echo "--- 测试 2: 写代码 ---"
ollama run "${SMALL_MODEL}" "写一个 Python 快速排序函数，不超过 10 行。" <<< ""
echo ""

echo "--- 测试 3: 翻译 ---"
ollama run "${SMALL_MODEL}" "Translate to Chinese: The open-source revolution made AI accessible to everyone." <<< ""
echo ""

# --------------------------------------------------
# 3. 用 API 调用（REST）
# --------------------------------------------------
echo ">>> 步骤 3: REST API 调用 ..."
echo ""

# 确保 Ollama 服务在运行
# （ollama run 会自动启动服务，但显式检查更稳妥）
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[INFO] 启动 Ollama 服务 ..."
    ollama serve &
    sleep 3
fi

echo "--- API: 生成回复 ---"
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"${SMALL_MODEL}\",
  \"prompt\": \"什么是 Transformer？用一句话回答。\",
  \"stream\": false
}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','[no response]'))" 2>/dev/null \
  || echo "[INFO] python3 不可用，跳过 JSON 解析"
echo ""

# --------------------------------------------------
# 4. 列出本地已有模型
# --------------------------------------------------
echo ">>> 步骤 4: 本地模型列表 ..."
ollama list
echo ""

# --------------------------------------------------
# 5.（可选）拉取大模型（LLaMA 3.1-70B Q4）
#    需要 ~48 GB RAM，取消注释以启用
# --------------------------------------------------
# LARGE_MODEL="llama3.1:70b-instruct-q4_K_M"
# echo ">>> 步骤 5: 拉取大模型 ${LARGE_MODEL} (~40 GB) ..."
# ollama pull "${LARGE_MODEL}"
# echo ""
# echo "--- 70B 测试: 复杂推理 ---"
# ollama run "${LARGE_MODEL}" "证明根号 2 是无理数，给出简洁的反证法。" <<< ""
# echo ""

# --------------------------------------------------
# 6. 清理（可选）
# --------------------------------------------------
# echo ">>> 清理: 删除模型以释放磁盘 ..."
# ollama rm "${SMALL_MODEL}"
# ollama rm "${LARGE_MODEL}"

echo "=============================================="
echo " Demo 完成！"
echo " "
echo " 关键收获："
echo "   1. 一条命令拉取并运行量化模型"
echo "   2. 所有推理在本地完成，数据不出机器"
echo "   3. REST API 可集成到任何应用"
echo "=============================================="
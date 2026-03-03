# Embedding 训练 Demo（纯 Python，无需额外依赖）

这个小项目用于快速上手 `embedding` 训练，目标是：

- 从零训练一个词向量（word embedding）
- 直接在当前仓库文本/代码上训练（不需要下载数据集）
- 跑一个可见结果的 demo（查询相似词）

## 项目思路（入门版）

我们用的是经典的 `Skip-gram + Negative Sampling (SGNS)`，这是 `word2vec` 的核心训练思路之一。

训练流程：

1. 扫描仓库里的 `*.py / *.md / *.tex` 文件作为语料
2. 分词（这里用英文/代码 token，例如 `rmha`, `policy`, `reward`）
3. 构建词表（按频次过滤）
4. 生成 `(中心词, 上下文词)` 训练样本
5. 用负采样做 SGD 训练
6. 保存词向量并查询相似词

## 为什么这个方案适合你（第一次训练 embedding）

- 不依赖 `torch` / `numpy`，环境更容易跑通
- 代码量小，能看懂训练细节（点积、sigmoid、负采样）
- 结果可解释：你能直接看到 `rmha`、`mappo`、`policy` 等词的邻居

## 运行方式

在仓库根目录执行：

```bash
python3 embedding_demo/train_repo_embedding.py \
  --steps 12000 \
  --dim 24 \
  --window 2 \
  --negatives 5 \
  --max-vocab 500 \
  --min-count 3
```

训练完成后查询相似词：

```bash
python3 embedding_demo/query_embedding.py --word rmha --topk 8
python3 embedding_demo/query_embedding.py --word mappo policy --topk 8
```

## 常用调参建议（先跑通再优化）

- `--steps`: 训练步数，先 `8000~20000`
- `--dim`: embedding 维度，先 `16/24/32`
- `--window`: 上下文窗口，代码语料通常 `2~4`
- `--negatives`: 负样本数，先 `3~8`
- `--min-count`: 过滤低频词，先 `2~5`
- `--max-vocab`: 词表上限，先 `300~1000`

## 下一步怎么升级（你熟悉后）

1. 换成 `numpy`/`torch` 实现（速度更快）
2. 用更干净的语料（只保留文档或只保留代码注释）
3. 增加 `subsampling`（高频词下采样）
4. 做可视化（PCA / t-SNE）
5. 训练句向量（sentence embedding）做检索 demo

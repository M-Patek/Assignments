# HS-ICS 匹配引擎 (Retrieve & Re-rank)

这是一个基于深度学习的自动化工具，旨在将 **HS Code（海关编码）** 描述通过语义分析精准匹配到 **ICS（国际标准分类）** 描述。

## 🌟 核心特性
* **双阶段匹配架构**：
    * **Stage 1 (快速召回)**：利用 `all-MiniLM-L6-v2` 计算语义相似度，并结合 `TF-IDF` 关键词修正，从海量库中初筛候选集。
    * **Stage 2 (精细重排序)**：使用 `Cross-Encoder` 深度模型对候选集进行二度打分，捕捉细微的词义差别。
* **上下文增强**：自动提取 HS 编码的章节（Chapter）和标题（Heading）信息，为底层的具体产品提供更丰富的描述背景。
* **鲁棒读取引擎**：具备“透视眼”功能，能够自动识别被误改后缀的 Excel/CSV 文件，并尝试多种字符编码（如 GB18030）以防乱码。

## 🛠️ 部署流程

### 1. 环境准备
确保您的系统中安装了 Python 3.8 或更高版本。

### 2. 安装依赖库
在终端中运行以下命令安装必要的库：
```bash
pip install pandas numpy scikit-learn sentence-transformers torch openpyxl
```

### 3. 模型预下载
代码在首次运行时会自动从 Hugging Face 下载以下预训练模型：
* `all-MiniLM-L6-v2` (Bi-Encoder)
* `ms-marco-MiniLM-L-6-v2` (Cross-Encoder)
*注：如果服务器无法连接外网，请提前下载模型并修改代码中的加载路径。*

### 4. 准备数据文件
将您的数据文件放置在与脚本相同的目录下，并确保文件名匹配：
* `HS07.xlsx - Sheet1.csv`
* `Data_ics_ed7.xlsx - Sheet1.csv`

### 5. 运行程序
```bash
python match_hs_to_ics.py
```

## ⚠️ 重要提示（主人请注意喵！）

本代码目前处于**硬编码配置阶段**，在正式部署前请根据实际需求修改代码开头的 `⚙️ 究极版配置` 区域：

1.  **文件路径硬编码**：
    * `HS_FILE_PATH` 和 `ICS_FILE_PATH` 当前指向了特定的文件名。如果您的文件名不同，请务必在代码中修改。
    * `OUTPUT_FILE` 默认为 `HS_to_ICS_Ultimate_Match.xlsx`。
2.  **超参数调整**：
    * `CANDIDATE_POOL_SIZE` (当前为 20)：决定了进入第二阶段重排序的候选数量。数值越大越精准，但耗时越长。
    * `ALPHA_SEMANTIC` (0.7) 与 `ALPHA_KEYWORD` (0.3)：控制语义和关键词权重的占比。
3.  **硬件要求**：
    * 代码使用 `torch` 运行模型。如果有显卡（CUDA），重排序阶段会大大加快；若无显卡则自动切换至 CPU 模式。

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

# ==========================================
# ⚙️ 究极版配置
# ==========================================
HS_FILE_PATH = "HS07.xlsx - Sheet1.csv"
ICS_FILE_PATH = "Data_ics_ed7.xlsx - Sheet1.csv"
OUTPUT_FILE = "HS_to_ICS_Ultimate_Match.csv"

# 初筛数量：第一步先找回多少个候选？
CANDIDATE_POOL_SIZE = 20  
# 最终保留数量
FINAL_TOP_K = 3           

# 权重配置（初筛阶段）
ALPHA_SEMANTIC = 0.7 
ALPHA_KEYWORD = 0.3

def preprocess_hs_with_context(hs_df):
    """
    上下文增强：将 HS 的章节标题 (2位编码) 拼接到 6位编码描述前。
    解决很多 6位编码描述只是 "Other" 或 "Parts" 的问题。
    """
    print("✨ 正在进行上下文增强处理...")
    hs_df['HS_Code'] = hs_df['HS_Code'].str.strip()
    hs_df['HS_Description'] = hs_df['HS_Description'].fillna('').str.strip()
    
    # 提取 2位数 章节 (Chapter) 及其描述
    # 假设 2位数的行其 HS_Code 长度为 2
    chapters = hs_df[hs_df['HS_Code'].str.len() == 2].set_index('HS_Code')['HS_Description'].to_dict()
    
    # 提取 4位数 (Heading) 及其描述 (可选，为了更精准可以加上)
    headings = hs_df[hs_df['HS_Code'].str.len() == 4].set_index('HS_Code')['HS_Description'].to_dict()

    # 筛选目标 6位数产品
    hs_target = hs_df[hs_df['HS_Code'].str.len() == 6].copy()
    
    enhanced_descriptions = []
    for idx, row in hs_target.iterrows():
        code = row['HS_Code']
        desc = row['HS_Description']
        
        # 查找父级
        chap_code = code[:2]
        head_code = code[:4]
        
        context_str = ""
        if chap_code in chapters:
            context_str += f"{chapters[chap_code]} > "
        if head_code in headings:
            # 有些 heading 描述太长，可以截断，这里简化处理直接拼接
            context_str += f"{headings[head_code]} > "
            
        # 拼接最终描述： [章节] > [标题] > [子目]
        full_desc = f"{context_str}{desc}"
        enhanced_descriptions.append(full_desc)
        
    hs_target['Enhanced_Description'] = enhanced_descriptions
    return hs_target

def run_ultimate_matching():
    print("🐱 启动究极匹配引擎 (Retrieve & Re-rank)...")
    
    # 1. 加载数据
    hs_df = pd.read_csv(HS_FILE_PATH, header=None, dtype=str)
    ics_df = pd.read_csv(ICS_FILE_PATH, header=None, dtype=str)
    
    hs_df.columns = ['HS_Code', 'HS_Description']
    ics_df.columns = ['ICS_Code', 'ICS_Description', 'Finest_Level']
    
    # 2. 上下文增强 (HS端)
    hs_target = preprocess_hs_with_context(hs_df)
    
    # ICS端处理
    ics_df['ICS_Description'] = ics_df['ICS_Description'].fillna('').str.strip()
    ics_target = ics_df[ics_df['Finest_Level'] == '1'].reset_index(drop=True)
    
    print(f"📊 待匹配 HS条目: {len(hs_target)} (已增强上下文)")
    print(f"📚 目标 ICS库: {len(ics_target)}")

    # ==========================================
    # 🚀 Stage 1: 快速召回 (Bi-Encoder + TF-IDF)
    # ==========================================
    print("\n[Stage 1] 快速召回 (Bi-Encoder)...")
    
    # 加载 Bi-Encoder 模型
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 编码
    # 注意：这里使用 'Enhanced_Description' 进行匹配，信息量更大
    hs_embeddings = bi_encoder.encode(hs_target['Enhanced_Description'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    ics_embeddings = bi_encoder.encode(ics_target['ICS_Description'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    # 语义相似度
    semantic_sim = cosine_similarity(hs_embeddings.cpu(), ics_embeddings.cpu())
    
    # TF-IDF 辅助 (针对关键词)
    print("[Stage 1] 关键词修正 (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english')
    # 训练集包含两者
    corpus = hs_target['Enhanced_Description'].tolist() + ics_target['ICS_Description'].tolist()
    tfidf.fit(corpus)
    
    hs_tfidf = tfidf.transform(hs_target['Enhanced_Description'])
    ics_tfidf = tfidf.transform(ics_target['ICS_Description'])
    keyword_sim = cosine_similarity(hs_tfidf, ics_tfidf)
    
    # 混合分数
    stage1_scores = (semantic_sim * ALPHA_SEMANTIC) + (keyword_sim * ALPHA_KEYWORD)

    # ==========================================
    # 💎 Stage 2: 精细重排序 (Cross-Encoder)
    # ==========================================
    print("\n[Stage 2] 深度重排序 (Cross-Encoder)... 此步骤较慢，但最准！")
    
    # 加载 Cross-Encoder 模型
    # ms-marco-MiniLM-L-6-v2 是专门训练来判断 "这两个句子是否相关" 的模型
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    results = []
    
    # 遍历每个 HS 产品
    total = len(hs_target)
    for i in range(total):
        # 1. 获取 Stage 1 分数最高的 Top N 个候选者的索引
        # argsort是从小到大，取最后 pool_size 个，再反转
        candidate_indices = stage1_scores[i].argsort()[-CANDIDATE_POOL_SIZE:][::-1]
        
        hs_text = hs_target.iloc[i]['Enhanced_Description']
        
        # 2. 准备 Cross-Encoder 的输入对
        # 格式: [[HS文本, ICS候选1], [HS文本, ICS候选2], ...]
        pairs = []
        valid_indices = [] # 记录对应的 ICS 索引
        
        for idx in candidate_indices:
            ics_text = ics_target.iloc[idx]['ICS_Description']
            pairs.append([hs_text, ics_text])
            valid_indices.append(idx)
            
        # 3. Cross-Encoder 打分 (预测 Logits)
        rerank_scores = cross_encoder.predict(pairs)
        
        # 4. 排序最终结果
        # 将分数和对应的 ICS 索引打包
        scored_candidates = list(zip(valid_indices, rerank_scores))
        # 按分数降序排列
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 5. 提取前 Top K
        top_k_matches = scored_candidates[:FINAL_TOP_K]
        
        match_strs = []
        for idx, score in top_k_matches:
            # Sigmoid 将 logit 转为 0-1 的概率感 (可选，这里直接用 raw score 也行)
            # 为了直观，我们只展示 Code 和 Desc
            ics_row = ics_target.iloc[idx]
            match_strs.append(f"[{ics_row['ICS_Code']}] {ics_row['ICS_Description']}")
            
        results.append({
            'HS_Code': hs_target.iloc[i]['HS_Code'],
            'HS_Description': hs_target.iloc[i]['HS_Description'], # 原始描述
            'Context_Used': hs_text, # 增强后的描述 (方便核对)
            'Best_Matches': " | ".join(match_strs)
        })
        
        if (i + 1) % 100 == 0:
            print(f"已精修 {i + 1}/{total} 个产品...")

    # ==========================================
    # 💾 保存结果
    # ==========================================
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ 究极匹配完成！文件已保存: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_ultimate_matching()

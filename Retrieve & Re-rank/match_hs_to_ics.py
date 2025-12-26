import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import os
import re

# ==========================================
# ⚙️ 究极版配置
# ==========================================
HS_FILE_PATH = "HS07.xlsx - Sheet1.csv"
ICS_FILE_PATH = "Data_ics_ed7.xlsx - Sheet1.csv"
# 修改：直接输出为 xlsx 格式，方便主人查看喵！
OUTPUT_FILE = "HS_to_ICS_Ultimate_Match.xlsx"

# 初筛数量：第一步先找回多少个候选？
CANDIDATE_POOL_SIZE = 20  
# 最终保留数量
FINAL_TOP_K = 3           

# 权重配置（初筛阶段）
ALPHA_SEMANTIC = 0.7 
ALPHA_KEYWORD = 0.3

def robust_read_csv(file_path):
    """
    自动尝试多种编码 + 多种分隔符读取 CSV 的聪明小工具喵
    (升级版：优先寻找列数 >= 2 的结果)
    """
    # 常见编码
    encodings = ['utf-8', 'gbk', 'utf-16', 'latin1']
    # 常见分隔符：逗号，Tab，分号
    separators = [',', '\t', ';']
    
    print(f"🔍 正在尝试读取文件: {file_path} ...")
    
    best_df = None
    max_cols = 0
    
    for enc in encodings:
        for sep in separators:
            try:
                # engine='python' 对错误的处理稍微宽容一点
                df = pd.read_csv(
                    file_path, 
                    header=None, 
                    dtype=str, 
                    encoding=enc, 
                    sep=sep,
                    on_bad_lines='skip', 
                    engine='python'
                )
                
                # 如果现在的列数比之前试出来的都多，就暂存这个结果
                if df.shape[1] > max_cols:
                    max_cols = df.shape[1]
                    best_df = df
                    
                # 如果找到了 >= 2 列的数据，这很可能就是对的，直接返回！
                if df.shape[1] >= 2:
                    print(f"✅ 成功读取! 编码: {enc}, 分隔符: {repr(sep)}, 形状: {df.shape}")
                    return df
                
            except Exception:
                # 如果报错了，就默默尝试下一个组合
                continue
    
    # 如果试了一圈还是找不到 >= 2 列的，就返回列数最多的那个（虽然可能只有1列）
    if best_df is not None:
        print(f"⚠️ 警告: 猫猫没能找到完美的格式，使用的是: 形状 {best_df.shape}。尝试后续修复...")
        return best_df
                
    raise ValueError(f"🙀 呜呜，猫猫用尽全力也没能读懂这个文件的格式: {file_path}")

def fix_one_column_df(df, name="数据"):
    """
    如果数据只有1列，尝试智能修复
    """
    if df.shape[1] >= 2:
        return df
        
    print(f"🔧 {name} 只有1列，猫猫尝试进行智能分列...")
    
    # 尝试 1: 用逗号或分号拆分第一列
    # 假设格式是 "Code;Description" 但没被正确解析
    try:
        series = df.iloc[:, 0].astype(str)
        # 尝试常见分隔符拆分 (expand=True 会变成多列)
        for sep in [',', ';', '\t', ' ']:
            split_df = series.str.split(sep, n=1, expand=True)
            if split_df.shape[1] >= 2:
                print(f"   ✨ 使用 '{sep}' 成功拆分!")
                return split_df
    except Exception:
        pass
        
    print(f"   💨 拆分失败，将复制第一列作为第二列以防崩溃...")
    # 实在不行，就复制一列，防止程序崩溃 (虽然结果可能不太对)
    df['Description_Placeholder'] = df.iloc[:, 0]
    return df

def preprocess_hs_with_context(hs_df):
    """
    上下文增强：将 HS 的章节标题 (2位编码) 拼接到 6位编码描述前。
    解决很多 6位编码描述只是 "Other" 或 "Parts" 的问题。
    """
    print("✨ 正在进行上下文增强处理...")
    
    # 确保至少有两列
    if hs_df.shape[1] < 2:
        hs_df = fix_one_column_df(hs_df, "HS数据")
        
    # 强制取前两列
    hs_df = hs_df.iloc[:, :2]
    hs_df.columns = ['HS_Code', 'HS_Description']
    
    # === 新增：强力清洗逻辑 ===
    # 1. 转为字符串
    hs_df['HS_Code'] = hs_df['HS_Code'].astype(str)
    # 2. 去除小数点 (如 1234.56 -> 123456)
    hs_df['HS_Code'] = hs_df['HS_Code'].str.replace('.', '', regex=False)
    # 3. 去除前后空格
    hs_df['HS_Code'] = hs_df['HS_Code'].str.strip()
    
    hs_df['HS_Description'] = hs_df['HS_Description'].fillna('').str.strip()
    
    # 提取 2位数 章节 (Chapter) 及其描述
    chapters = hs_df[hs_df['HS_Code'].str.len() == 2].set_index('HS_Code')['HS_Description'].to_dict()
    
    # 提取 4位数 (Heading) 及其描述
    headings = hs_df[hs_df['HS_Code'].str.len() == 4].set_index('HS_Code')['HS_Description'].to_dict()

    # 筛选目标 6位数产品 (包括原本大于6位的截取前6位)
    # 只要长度 >= 6 的都保留
    hs_target = hs_df[hs_df['HS_Code'].str.len() >= 6].copy()
    
    # 如果是8位或更多，截取前6位作为标准 HS6
    hs_target['HS6_Clean'] = hs_target['HS_Code'].str.slice(0, 6)
    
    enhanced_descriptions = []
    for idx, row in hs_target.iterrows():
        code = row['HS6_Clean'] # 使用清洗后的6位码找父级
        desc = row['HS_Description']
        
        # 安全切片
        chap_code = str(code)[:2]
        head_code = str(code)[:4]
        
        context_str = ""
        if chap_code in chapters:
            context_str += f"{chapters[chap_code]} > "
        if head_code in headings:
            context_str += f"{headings[head_code]} > "
            
        full_desc = f"{context_str}{desc}"
        enhanced_descriptions.append(full_desc)
        
    hs_target['Enhanced_Description'] = enhanced_descriptions
    return hs_target

def run_ultimate_matching():
    print("🐱 启动究极匹配引擎 (Retrieve & Re-rank)...")
    
    # 1. 加载数据 (使用鲁棒读取)
    hs_df = robust_read_csv(HS_FILE_PATH)
    ics_df = robust_read_csv(ICS_FILE_PATH)
    
    # 检查并修复列数
    if ics_df.shape[1] < 2:
        ics_df = fix_one_column_df(ics_df, "ICS数据")
    
    # ICS端处理
    # 确保 ICS 至少有3列，如果没有，尝试兼容
    if ics_df.shape[1] >= 3:
         ics_df = ics_df.iloc[:, :3]
         ics_df.columns = ['ICS_Code', 'ICS_Description', 'Finest_Level']
    else:
        print("⚠️ 警告：ICS 文件少于3列，猫猫尝试强制解析...")
        # 只有2列的情况
        ics_df = ics_df.iloc[:, :2]
        ics_df.columns = ['ICS_Code', 'ICS_Description']
        ics_df['Finest_Level'] = '1' # 假定全是细分级别
    
    # 2. 上下文增强 (HS端)
    try:
        hs_target = preprocess_hs_with_context(hs_df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 处理 HS 数据时出错: {e}")
        return
    
    # ICS 数据清洗
    ics_df['ICS_Description'] = ics_df['ICS_Description'].fillna('').str.strip()
    # 筛选细分级别为 1 的条目
    ics_target = ics_df[ics_df['Finest_Level'] == '1'].reset_index(drop=True)
    
    # 如果筛选后为空，可能是 Finest_Level 列不对，尝试放宽条件
    if len(ics_target) == 0:
        print("⚠️ 警告：筛选 Finest_Level='1' 后为空，尝试使用所有 ICS 条目...")
        ics_target = ics_df
    
    if len(ics_target) == 0:
        print("❌ 错误：ICS 目标库为空，无法进行匹配喵！")
        return

    print(f"📊 待匹配 HS条目: {len(hs_target)} (已增强上下文)")
    print(f"📚 目标 ICS库: {len(ics_target)}")

    # ==========================================
    # 🚀 Stage 1: 快速召回 (Bi-Encoder + TF-IDF)
    # ==========================================
    print("\n[Stage 1] 快速召回 (Bi-Encoder)...")
    
    # 加载 Bi-Encoder 模型
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 编码
    hs_descriptions = hs_target['Enhanced_Description'].tolist()
    ics_descriptions = ics_target['ICS_Description'].tolist()
    
    # 检查是否为空
    if not hs_descriptions:
        print("🙀 哎呀，HS 数据列表为空，可能是清洗过程把所有数据都过滤掉了喵！")
        return

    hs_embeddings = bi_encoder.encode(hs_descriptions, convert_to_tensor=True, show_progress_bar=True)
    ics_embeddings = bi_encoder.encode(ics_descriptions, convert_to_tensor=True, show_progress_bar=True)
    
    # 语义相似度 (余弦相似度)
    semantic_sim = cosine_similarity(hs_embeddings.cpu(), ics_embeddings.cpu())
    
    # TF-IDF 辅助 (针对硬核关键词)
    print("[Stage 1] 关键词修正 (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = hs_descriptions + ics_descriptions
    tfidf.fit(corpus)
    
    hs_tfidf = tfidf.transform(hs_descriptions)
    ics_tfidf = tfidf.transform(ics_descriptions)
    keyword_sim = cosine_similarity(hs_tfidf, ics_tfidf)
    
    # 混合分数
    stage1_scores = (semantic_sim * ALPHA_SEMANTIC) + (keyword_sim * ALPHA_KEYWORD)

    # ==========================================
    # 💎 Stage 2: 精细重排序 (Cross-Encoder)
    # ==========================================
    print("\n[Stage 2] 深度重排序 (Cross-Encoder)... 此步骤计算量较大，猫猫正在全力以赴喵！")
    
    # 加载 Cross-Encoder 模型
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    results = []
    total = len(hs_target)
    
    # 遍历每个 HS 产品进行精细打分
    for i in range(total):
        # 获取 Stage 1 分数最高的一批候选者
        # argsort 从小到大，取最后 CANDIDATE_POOL_SIZE 个并反转
        candidate_indices = stage1_scores[i].argsort()[-CANDIDATE_POOL_SIZE:][::-1]
        
        hs_text = hs_target.iloc[i]['Enhanced_Description']
        
        # 准备 Cross-Encoder 的输入对
        pairs = []
        valid_indices = []
        
        for idx in candidate_indices:
            ics_text = ics_target.iloc[idx]['ICS_Description']
            pairs.append([hs_text, ics_text])
            valid_indices.append(idx)
            
        # 打分
        rerank_scores = cross_encoder.predict(pairs)
        
        # 排序
        scored_candidates = sorted(zip(valid_indices, rerank_scores), key=lambda x: x[1], reverse=True)
        
        # 提取 Top K
        top_k_matches = scored_candidates[:FINAL_TOP_K]
        
        match_strs = []
        for idx, score in top_k_matches:
            ics_row = ics_target.iloc[idx]
            match_strs.append(f"[{ics_row['ICS_Code']}] {ics_row['ICS_Description']}")
            
        results.append({
            'HS_Code': hs_target.iloc[i]['HS_Code'],
            'HS_Description': hs_target.iloc[i]['HS_Description'],
            'Context_Used': hs_text,
            'Best_Matches': " | ".join(match_strs)
        })
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{total} 个产品...")

    # ==========================================
    # 💾 保存结果
    # ==========================================
    df_res = pd.DataFrame(results)
    # 修改：直接保存为 Excel
    print(f"💾 正在保存为 Excel 文件: {OUTPUT_FILE} ...")
    df_res.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"\n✅ 究极匹配完成！主人喵，请查看文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        run_ultimate_matching()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"🙀 哎呀，运行出错了主人喵: {e}")

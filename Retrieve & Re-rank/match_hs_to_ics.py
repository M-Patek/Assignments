import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import os
import re

HS_FILE_PATH = "HS07.xlsx - Sheet1.csv"
ICS_FILE_PATH = "Data_ics_ed7.xlsx - Sheet1.csv"
OUTPUT_FILE = "HS_to_ICS_Ultimate_Match.xlsx"

# 初筛数量
CANDIDATE_POOL_SIZE = 20  
# 最终保留数量
FINAL_TOP_K = 3           

# 权重配置
ALPHA_SEMANTIC = 0.7 
ALPHA_KEYWORD = 0.3

def robust_read_csv(file_path):
    """
    自动尝试多种编码 + 多种分隔符读取 CSV 的工具
    (升级版：增加了“透视眼”，专门处理把 xlsx 直接改名为 csv 的情况！)
    """
    print(f"🔍 正在尝试读取文件: {file_path} ...")
    
    # === 0. 优先尝试：这是否是伪装成 CSV 的 Excel 文件？ ===
    # 主人说直接改了后缀，所以这其实是 Excel (zip) 文件！
    try:
        # 我们用二进制方式打开，绕过后缀名检查，直接喂给 read_excel
        with open(file_path, 'rb') as f:
            df = pd.read_excel(f, header=None, dtype=str, engine='openpyxl')
            
        print(f"✅ 发现这其实是一个 Excel 文件！成功读取: 形状 {df.shape}")
        return df
    except Exception as e:
        # 如果不是 Excel，或者读取失败，就继续往下走
        # print(f"   (并不是 Excel 文件，继续尝试文本模式...)")
        pass

    # === 如果上面失败了，说明它真的是个文本文件，开始尝试各种编码 ===
    
    # 优先级调整：GB18030 (中文) 排在最前面！
    strict_encodings = ['gb18030', 'utf-8-sig', 'utf-8', 'gbk']
    separators = [';', ',', '\t'] 
    
    best_df = None
    max_cols = 0
    
    # === 第一轮：严格模式 (Strict) ===
    for enc in strict_encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    file_path, 
                    header=None, 
                    dtype=str, 
                    encoding=enc, 
                    sep=sep,
                    on_bad_lines='skip', 
                    engine='python'
                )
                if df.shape[1] > max_cols:
                    max_cols = df.shape[1]
                    best_df = df
                    best_df.attrs['encoding_used'] = enc 
                
                if df.shape[1] >= 2:
                    print(f"✅ [严格模式] 成功读取! 编码: {enc}, 分隔符: {repr(sep)}, 形状: {df.shape}")
                    return df
            except Exception:
                continue

    # === 第二轮：容错模式 (Replace) ===
    print("⚠️ 严格模式读取失败，猫猫开启容错模式（忽略个别坏字符）...")
    for enc in ['gb18030', 'utf-8-sig', 'latin1']:
        for sep in separators:
            try:
                df = pd.read_csv(
                    file_path, 
                    header=None, 
                    dtype=str, 
                    encoding=enc, 
                    sep=sep,
                    encoding_errors='replace',
                    on_bad_lines='skip', 
                    engine='python'
                )
                if df.shape[1] > max_cols:
                    max_cols = df.shape[1]
                    best_df = df
                
                if df.shape[1] >= 2:
                    print(f"✅ [容错模式] 成功读取! 编码: {enc}, 分隔符: {repr(sep)}, 形状: {df.shape}")
                    return df
            except Exception:
                continue

    # === 第三轮：最后手段 ===
    if best_df is not None:
        print(f"⚠️ 警告: 使用了不太完美的读取方式 (可能含乱码)，形状: {best_df.shape}")
        return best_df

    raise ValueError(f"没能读懂这个文件的格式: {file_path}")

def fix_one_column_df(df, name="数据"):
    """
    如果数据只有1列，尝试智能修复
    """
    if df.shape[1] >= 2:
        return df
        
    print(f"🔧 {name} 只有1列，猫猫尝试进行智能分列...")
    try:
        series = df.iloc[:, 0].astype(str)
        # 常见分隔符
        for sep in [';', ',', '\t', ' ']:
            split_df = series.str.split(sep, n=1, expand=True)
            if split_df.shape[1] >= 2:
                print(f"   ✨ 使用 '{sep}' 成功拆分!")
                return split_df
    except Exception:
        pass
        
    print(f"   💨 拆分失败，将复制第一列作为第二列以防崩溃...")
    df['Description_Placeholder'] = df.iloc[:, 0]
    return df

def preprocess_hs_with_context(hs_df):
    """
    上下文增强处理
    """
    print("✨ 正在进行上下文增强处理...")
    
    if hs_df.shape[1] < 2:
        hs_df = fix_one_column_df(hs_df, "HS数据")
        
    hs_df = hs_df.iloc[:, :2]
    hs_df.columns = ['HS_Code', 'HS_Description']
    
    # 强力清洗
    hs_df['HS_Code'] = hs_df['HS_Code'].astype(str).str.replace('.', '', regex=False).str.strip()
    hs_df['HS_Description'] = hs_df['HS_Description'].fillna('').str.strip()
    
    # 提取章节和标题
    chapters = hs_df[hs_df['HS_Code'].str.len() == 2].set_index('HS_Code')['HS_Description'].to_dict()
    headings = hs_df[hs_df['HS_Code'].str.len() == 4].set_index('HS_Code')['HS_Description'].to_dict()

    # 只要长度 >= 6 的都保留
    hs_target = hs_df[hs_df['HS_Code'].str.len() >= 6].copy()
    hs_target['HS6_Clean'] = hs_target['HS_Code'].str.slice(0, 6)
    
    enhanced_descriptions = []
    for idx, row in hs_target.iterrows():
        code = row['HS6_Clean']
        desc = row['HS_Description']
        
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
    print("🐱 启动匹配引擎 (Retrieve & Re-rank)...")
    
    # 1. 加载数据
    hs_df = robust_read_csv(HS_FILE_PATH)
    ics_df = robust_read_csv(ICS_FILE_PATH)
    
    # 修复列数
    if ics_df.shape[1] < 2:
        ics_df = fix_one_column_df(ics_df, "ICS数据")
    
    if ics_df.shape[1] >= 3:
         ics_df = ics_df.iloc[:, :3]
         ics_df.columns = ['ICS_Code', 'ICS_Description', 'Finest_Level']
    else:
        print("⚠️ 警告：ICS 文件少于3列，尝试强制解析...")
        ics_df = ics_df.iloc[:, :2]
        ics_df.columns = ['ICS_Code', 'ICS_Description']
        ics_df['Finest_Level'] = '1' 
    
    # 2. 上下文增强
    try:
        hs_target = preprocess_hs_with_context(hs_df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 处理 HS 数据时出错: {e}")
        return
    
    # ICS 清洗
    ics_df['ICS_Description'] = ics_df['ICS_Description'].fillna('').str.strip()
    ics_target = ics_df[ics_df['Finest_Level'] == '1'].reset_index(drop=True)
    
    if len(ics_target) == 0:
        print("⚠️ 警告：筛选 Finest_Level='1' 后为空，尝试使用所有 ICS 条目...")
        ics_target = ics_df
    
    if len(ics_target) == 0:
        print("❌ 错误：ICS 目标库为空，无法进行匹配！")
        return

    print(f"📊 待匹配 HS条目: {len(hs_target)}")
    print(f"📚 目标 ICS库: {len(ics_target)}")

    # ==========================================
    # Stage 1: 快速召回
    # ==========================================
    print("\n[Stage 1] 快速召回 (Bi-Encoder)...")
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    hs_descriptions = hs_target['Enhanced_Description'].tolist()
    ics_descriptions = ics_target['ICS_Description'].tolist()
    
    if not hs_descriptions:
        print("🙀 HS 数据列表为空！")
        return

    hs_embeddings = bi_encoder.encode(hs_descriptions, convert_to_tensor=True, show_progress_bar=True)
    ics_embeddings = bi_encoder.encode(ics_descriptions, convert_to_tensor=True, show_progress_bar=True)
    
    semantic_sim = cosine_similarity(hs_embeddings.cpu(), ics_embeddings.cpu())
    
    print("[Stage 1] 关键词修正 (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = hs_descriptions + ics_descriptions
    tfidf.fit(corpus)
    
    hs_tfidf = tfidf.transform(hs_descriptions)
    ics_tfidf = tfidf.transform(ics_descriptions)
    keyword_sim = cosine_similarity(hs_tfidf, ics_tfidf)
    
    stage1_scores = (semantic_sim * ALPHA_SEMANTIC) + (keyword_sim * ALPHA_KEYWORD)

    # ==========================================
    # Stage 2: 精细重排序
    # ==========================================
    print("\n[Stage 2] 深度重排序 (Cross-Encoder)...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    results = []
    total = len(hs_target)
    
    for i in range(total):
        candidate_indices = stage1_scores[i].argsort()[-CANDIDATE_POOL_SIZE:][::-1]
        hs_text = hs_target.iloc[i]['Enhanced_Description']
        
        pairs = []
        valid_indices = []
        for idx in candidate_indices:
            ics_text = ics_target.iloc[idx]['ICS_Description']
            pairs.append([hs_text, ics_text])
            valid_indices.append(idx)
            
        rerank_scores = cross_encoder.predict(pairs)
        scored_candidates = sorted(zip(valid_indices, rerank_scores), key=lambda x: x[1], reverse=True)
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

    # 保存
    df_res = pd.DataFrame(results)
    
    # 清洗非法字符
    def clean_illegal_chars(text):
        if isinstance(text, str):
            # 移除不可见字符 (0-31)，保留 \t \n \r
            return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text

    try:
        df_res = df_res.map(clean_illegal_chars)
    except AttributeError:
        df_res = df_res.applymap(clean_illegal_chars)

    print(f"💾 正在保存为 Excel 文件: {OUTPUT_FILE} ...")
    try:
        df_res.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
        print(f"\n✅ 究极匹配完成！主人喵，请查看文件: {OUTPUT_FILE}")
    except Exception as e:
        print(f"⚠️ 保存 Excel 失败 ({e})，尝试保存 CSV...")
        csv_backup = OUTPUT_FILE.replace('.xlsx', '_backup.csv')
        df_res.to_csv(csv_backup, index=False, encoding='utf-8-sig')
        print(f"✅ 已紧急保存为 CSV: {csv_backup}")

if __name__ == "__main__":
    try:
        run_ultimate_matching()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"🙀 哎呀，运行出错了喵: {e}")

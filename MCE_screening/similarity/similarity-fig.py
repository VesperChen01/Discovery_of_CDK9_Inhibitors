from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import Draw, rdFMCS
#from IPython.display import display 

# 读取excel文件
df = pd.read_excel('/home/roufen/crf/ML_test/CDK9/MCE_screening/similarity/CDK9vsMCE-orismi.xlsx')

# 获取分子smiles
query_smiles = df['Query_smiles'].dropna().values
data_smiles = df['Data_smiles'].dropna().unique()

# 计算所有Data_smiles的分子指纹
data_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2) for smi in data_smiles]

# 初始化最大相似度和最相似分子列
df['Max_similarity'] = 0.0
df['Max_sim_smiles'] = ''

num_queries = len(query_smiles)
num_data = len(data_smiles)


tanimoto_matrix = np.zeros((num_queries, num_data))

for i, q_smile in enumerate(query_smiles):
    # 检查q_smile是否为字符串
    if not isinstance(q_smile, str):
        print(f"Skipping non-string SMILES at index {i}: {q_smile}")
        continue

    # 创建分子指纹
    query_mol = Chem.MolFromSmiles(q_smile)
    if query_mol is None: 
        print(f"Invalid SMILES at index {i}: {q_smile}")
        continue  # 如果smiles无效，跳过

    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)

    # 计算Tanimoto系数
    tcs = DataStructs.BulkTanimotoSimilarity(query_fp, data_fps)
    
    # 找到最大的相似度及其索引
    max_sim_idx = np.argmax(tcs)
    max_similarity = tcs[max_sim_idx]

    df.loc[i, 'Max_similarity'] = max_similarity
    df.loc[i, 'Max_sim_smiles'] = data_smiles[max_sim_idx]
    tanimoto_matrix[i, :] = tcs
# 保存结果
df.to_excel('/home/roufen/crf/ML_test/CDK9/MCE_screening/similarity/CDK9vsMCE-resultsmi.xlsx')







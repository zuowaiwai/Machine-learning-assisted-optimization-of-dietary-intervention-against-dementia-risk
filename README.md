# UKBB GWAS Imputed v3 数据清单 - CSV 格式

本文件夹包含从 UK Biobank GWAS Imputed v3 File Manifest (Release 20180731) Excel文件转换而来的CSV文件。

## 📁 文件列表

### 1. **Manifest_201807.csv** (主要数据清单)
- **行数**: 11,934 行（其中11,912行为有效数据）
- **用途**: 这是最重要的文件，包含所有GWAS数据文件的完整清单和下载信息

#### 列说明：
- `Phenotype Code`: 表型代码（如：100001_irnt, 100001_raw, age, is_female等）
- `Phenotype Description`: 表型描述（如：Food weight, Energy, Protein等）
- `UK Biobank Data Showcase Link`: UK Biobank数据展示页面的链接
- `Sex`: 性别分类
  - `both_sexes` (4,587个文件)
  - `female` (3,742个文件)
  - `male` (3,583个文件)
- `File`: 数据文件名（格式：`{phenotype_code}.gwas.imputed_v3.{sex}.tsv.bgz`）
- `wget command`: 使用wget下载文件的完整命令
- `AWS File`: AWS S3存储的文件URL
- `Dropbox File`: Dropbox存储的文件URL
- `md5s`: 文件的MD5校验值，用于验证文件完整性

#### 使用示例：
```python
import pandas as pd

# 读取清单
df = pd.read_csv('Manifest_201807.csv')

# 过滤掉空行
df_valid = df[df['Phenotype Code'].notna()]

# 查找特定表型的数据
food_weight_data = df_valid[df_valid['Phenotype Description'].str.contains('Food weight', na=False)]

# 获取所有both_sexes的数据
both_sexes_data = df_valid[df_valid['Sex'] == 'both_sexes']

# 提取下载链接
aws_links = df_valid['AWS File'].tolist()
```

---

### 2. **Description_Lookup.csv** (表型描述查找表)
- **行数**: 11,372 行
- **唯一表型数**: 4,539 个
- **用途**: 快速查找表型代码对应的人类可读描述

#### 列说明：
- 第1列: 表型代码（如：100001_irnt, 100001_raw）
  - `_irnt` 后缀: Inverse rank normalized transformation（逆秩正态化转换）
  - `_raw` 后缀: 原始数据，未经转换
- 第2列: 表型描述（如：Food weight, Energy, Protein, Fat等）

#### 使用示例：
```python
import pandas as pd

# 读取查找表（注意：第一行是数据，不是列名）
df_lookup = pd.read_csv('Description_Lookup.csv', header=None, names=['Phenotype_Code', 'Description'])

# 查找特定代码的描述
code = '100001_irnt'
description = df_lookup[df_lookup['Phenotype_Code'] == code]['Description'].iloc[0]
print(f"{code}: {description}")

# 查找所有与"Energy"相关的表型
energy_phenotypes = df_lookup[df_lookup['Description'].str.contains('Energy', na=False)]

# 获取所有唯一的描述
unique_descriptions = df_lookup['Description'].unique()
print(f"共有 {len(unique_descriptions)} 个不同的表型类别")
```

---

### 3. **md5s_for_files.csv** (MD5校验值)
- **行数**: 11,516 行
- **用途**: 验证下载文件的完整性和准确性

#### 列说明：
- `file`: 数据文件名
- `md5 hex`: 文件的MD5哈希值（32位十六进制字符串）

#### 使用示例：
```python
import pandas as pd
import hashlib

# 读取MD5清单
df_md5 = pd.read_csv('md5s_for_files.csv')

# 验证下载文件的MD5值
def verify_file_md5(file_path, expected_md5):
    """验证文件的MD5校验值"""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    calculated_md5 = md5_hash.hexdigest()
    return calculated_md5 == expected_md5

# 查找特定文件的MD5值
filename = '100001_irnt.gwas.imputed_v3.both_sexes.tsv.bgz'
expected_md5 = df_md5[df_md5['file'] == filename]['md5 hex'].iloc[0]
print(f"文件 {filename} 的预期MD5: {expected_md5}")

# 验证文件
is_valid = verify_file_md5(f'/path/to/{filename}', expected_md5)
print(f"文件完整性验证: {'通过' if is_valid else '失败'}")
```

---

### 4. **README.csv** (原始说明文档)
- **行数**: 100 行
- **用途**: 包含原始Excel文件中的README说明
- **注意**: 这是从Excel转换而来的格式化文本，可能包含多个空列

---

### 5. **DEPRECATED___DROPBOX_Manifest_2.csv** (已弃用)
- **行数**: 11,941 行
- **状态**: ⚠️ 已弃用，不应使用
- **说明**: 这是旧版的Dropbox清单，数据已迁移到AWS

---

## 🔍 常见使用场景

### 场景1: 下载特定表型的所有性别数据
```python
import pandas as pd
import subprocess

df = pd.read_csv('Manifest_201807.csv')
df_valid = df[df['Phenotype Code'].notna()]

# 下载"Food weight"相关的所有数据
phenotype = 'Food weight'
files = df_valid[df_valid['Phenotype Description'] == phenotype]

for idx, row in files.iterrows():
    wget_cmd = row['wget command']
    print(f"正在下载: {row['File']}")
    subprocess.run(wget_cmd, shell=True)
```

### 场景2: 批量下载both_sexes数据
```python
import pandas as pd

df = pd.read_csv('Manifest_201807.csv')
df_valid = df[df['Phenotype Code'].notna()]

# 只下载both_sexes的数据
both_sexes = df_valid[df_valid['Sex'] == 'both_sexes']

# 生成下载脚本
with open('download_both_sexes.sh', 'w') as f:
    f.write('#!/bin/bash\n\n')
    for cmd in both_sexes['wget command'].dropna():
        f.write(f'{cmd}\n')

print(f"已生成下载脚本，包含 {len(both_sexes)} 个文件")
```

### 场景3: 关联表型代码和描述
```python
import pandas as pd

# 读取清单和查找表
df_manifest = pd.read_csv('Manifest_201807.csv')
df_lookup = pd.read_csv('Description_Lookup.csv', header=None, names=['Phenotype_Code', 'Description'])

# 合并数据
df_manifest_valid = df_manifest[df_manifest['Phenotype Code'].notna()]
df_merged = df_manifest_valid.merge(
    df_lookup,
    left_on='Phenotype Code',
    right_on='Phenotype_Code',
    how='left'
)

# 查看合并结果
print(df_merged[['Phenotype Code', 'Description', 'Sex', 'File']].head(20))
```

### 场景4: 验证下载文件的完整性
```python
import pandas as pd
import hashlib
import os

def calculate_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

# 读取MD5清单
df_md5 = pd.read_csv('md5s_for_files.csv')

# 验证下载目录中的所有文件
download_dir = '/path/to/downloaded/files'
for idx, row in df_md5.iterrows():
    file_path = os.path.join(download_dir, row['file'])
    if os.path.exists(file_path):
        calculated = calculate_md5(file_path)
        expected = row['md5 hex']
        status = '✓' if calculated == expected else '✗'
        print(f"{status} {row['file']}: {calculated == expected}")
```

---

## 📊 数据统计

- **总表型数**: 4,539个唯一表型
- **总文件数**: 11,912个GWAS数据文件
- **性别分类**:
  - both_sexes: 4,587个文件 (38.5%)
  - female: 3,742个文件 (31.4%)
  - male: 3,583个文件 (30.1%)

---

## 💡 提示

1. **数据源**: 所有数据已从Dropbox迁移至AWS S3，建议使用AWS链接下载
2. **文件格式**: 数据文件为`.tsv.bgz`格式（Tab分隔值，经过bgzip压缩）
3. **MD5验证**: 下载完成后务必验证MD5值以确保数据完整性
4. **表型命名**:
   - `_irnt`后缀表示经过逆秩正态化转换的数据
   - `_raw`后缀表示原始未转换的数据
5. **性别特异性分析**: 根据研究需求选择合适的性别分类数据

---

## 📚 相关资源

- [UK Biobank Official Website](https://www.ukbiobank.ac.uk/)
- [UK Biobank Data Showcase](https://biobank.ndph.ox.ac.uk/showcase/)
- AWS数据存储: `https://broad-ukb-sumstats-us-east-1.s3.amazonaws.com/`

---

## 📝 版本信息

- **原始文件**: UKBB GWAS Imputed v3 - File Manifest Release 20180731.xlsx
- **转换日期**: 2025年11月10日
- **转换格式**: CSV (UTF-8编码)
- **文件数量**: 5个CSV文件

---

## ❓ 常见问题

**Q: 如何选择_irnt还是_raw数据？**
A: `_irnt`数据经过逆秩正态化转换，适合需要正态分布假设的统计分析；`_raw`是原始数据，保留了原始分布特征。

**Q: 为什么有些表型只有both_sexes数据？**
A: 某些表型（如is_female）本身与性别相关，因此只提供合并的数据。

**Q: 如何高效下载大量文件？**
A: 建议使用并行下载工具（如aria2c）或编写脚本批量下载，并使用MD5验证确保下载完整性。

**Q: 文件大小有多大？**
A: 单个文件大小因表型而异，通常在几MB到几百MB之间。建议先下载少量文件测试后再批量下载。

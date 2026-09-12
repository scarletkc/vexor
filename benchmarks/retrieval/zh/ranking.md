# 排序策略的范围

集合的 off 策略按向量相似度排列过滤后的记录。hybrid 在过滤后的全部记录上融合向量与 BM25 排名。

bm25、flashrank 和 remote 只重排扩大的向量候选窗口。集合重排器接收候选记录的全文；文件重排使用路径与预览，二者不能混为一谈。

来源：docs/api/collections.md、docs/roadmap.md。

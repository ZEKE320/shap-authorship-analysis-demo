def para2sent(para: list[list[str]]) -> list[str]:
    """段落のリストを文章のリストに変換する"""
    return [word for sent in para for word in sent]

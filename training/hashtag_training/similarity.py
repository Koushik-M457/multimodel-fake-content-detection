from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def relevance_score(caption, hashtags):
    caption_emb = model.encode(caption, convert_to_tensor=True)
    hashtag_emb = model.encode(hashtags, convert_to_tensor=True)

    score = util.cos_sim(caption_emb, hashtag_emb).item()
    return score


from icrawler.builtin import BingImageCrawler

# -------- REAL IMAGES --------
print("Downloading REAL images...")
real_crawler = BingImageCrawler(
    storage={"root_dir": "data/image/train/real"}
)
real_crawler.crawl(
    keyword="real human face portrait",
    max_num=500
)

# -------- AI / FAKE IMAGES --------
print("Downloading AI images...")
fake_crawler = BingImageCrawler(
    storage={"root_dir": "data/image/train/fake"}
)
fake_crawler.crawl(
    keyword="AI generated face realistic",
    max_num=500
)

print("Download complete ")
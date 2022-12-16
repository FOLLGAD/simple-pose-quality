from bing_image_downloader import downloader
query = "person standing hands on hips pose"
downloader.download(query, limit=100,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
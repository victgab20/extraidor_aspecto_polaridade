import gdown

file_id = "1rJ8XEJh_-cRz5lSHR0PNzIi9HtJC6e8u"
url = f"https://drive.google.com/uc?id={file_id}"
output = "train_data.csv"
gdown.download(url, output, quiet=False)

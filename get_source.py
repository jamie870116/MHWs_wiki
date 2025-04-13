import os
import bs4
import requests
import re
from urllib.parse import urljoin


# get the html of the wiki page -eng_wiki
base_url = "https://monsterhunterwilds.wiki.fextralife.com/Monster+Hunter+Wilds+Wiki"
domain_prefix = "https://monsterhunterwilds.wiki.fextralife.com/"
visited = set()
page_data = []

# create the folder
os.makedirs("wiki_pages", exist_ok=True)
os.makedirs("wiki_pages_html", exist_ok=True)

def sanitize_filename(url):
    last_part = url.split("/")[-1]
    name = re.sub(r'[^a-zA-Z0-9_]', '_', last_part)
    return name[:50] + ".txt"

# get <div class="flex-main"> and save the text
def extract_text_from_page(url):
    try:
        res = requests.get(url, timeout=10)
        soup = bs4.BeautifulSoup(res.text, "html.parser")
        main_div = soup.find("div", class_="page-content")
        if not main_div:
            print(f"❌ not found fex-main：{url}")
            return

        page_text = main_div.get_text(separator="\n", strip=True)
        filename = sanitize_filename(url)
        filepath = os.path.join("wiki_pages", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page_text)

        page_data.append({"url": url, "text": page_text})
        print(f"✅ Saved：{filename}")
    except Exception as e:
        print(f"❌ Error：{url} | error msg：{e}")

# get <div class="flex-main"> and save the html
def extract_html_from_page(url):
    try:
        res = requests.get(url, timeout=10)
        soup = bs4.BeautifulSoup(res.text, "html.parser")
        main_div = soup.find("div", class_="page-content")
        if not main_div:
            print(f"❌ not found fex-main：{url}")
            return

        # store the html
        page_html = main_div.prettify() 
        filename = sanitize_filename(url).replace(".txt", ".html")
        filepath = os.path.join("wiki_pages_html", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page_html)

        page_data.append({"url": url, "html": page_html})
        print(f"✅ Saved： HTML：{filename}")
    except Exception as e:
        print(f"❌ Error：：{url} | error msg：{e}")


# get all subpages excluding images
def get_subpages_excluding_images(start_url):
    res = requests.get(start_url)
    soup = bs4.BeautifulSoup(res.text, "html.parser")

    for a in soup.find_all("a", href=True):
        full_url = urljoin(start_url, a["href"])

        if (full_url.startswith(domain_prefix)
            and full_url not in visited
            and not full_url.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
            and 'jpg' not in full_url):

            visited.add(full_url)
            extract_text_from_page(full_url)
            extract_html_from_page(full_url)


# execute
get_subpages_excluding_images(base_url)

print(f"✅ Saved：{len(page_data)} pages in wiki_pages/")

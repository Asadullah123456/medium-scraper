# main.py
import os
from medium_api import MediumAPI

graph_query = open('searchquery.qu', 'r').read()
search_keywords = ['food', 'fashion and beauty' , 'sports']
headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0',
}

def main():
    for keyword in search_keywords:
        count = 0
        folder_path = f'{keyword}_data'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        medium_api = MediumAPI(headers)
        dics = medium_api.search_posts(keyword, graph_query, results_limit=50, max_pages=10)
        
        for dic in dics:
            try:
                content = medium_api.get_post_content(dic['medium_url'])
                if content:
                    count += 1
                    file_path = os.path.join(folder_path, f'{count}.txt')
                    f = open(file_path, 'w', encoding='utf-8')
                    f.write(content)
                    f.flush()
                    f.close()

            except Exception as e:
                print(f"Error processing post: {e}")

if __name__ == "__main__":
    main()
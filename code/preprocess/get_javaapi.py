from bs4 import BeautifulSoup
import pandas as pd
import re


urls = []
url_num = 27

for i in range(1, url_num + 1):
    url = '../../data/java-api/index-files/index-' + \
        str(i) + '.html'
    urls.append(url)

index_name = []
index_description = []
method_description = []

for i in range(url_num):
    with open(urls[i], encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')
        content = soup.find_all(['dt', 'dd'])
        last_is_dd = True
        for i in range(len(content)):
            text = content[i].get_text()
            if '<dt>' in str(content[i]) and '</dt>' in str(content[i]):
                if not last_is_dd:
                    method_description.append('None')
                index_name.append(text.split(' - ')[0])
                index_description.append(text.split(' - ')[1])
                last_is_dd = False
            else:
                if last_is_dd:
                    index_name.append('None')
                    index_description.append('None')
                if text == '\xa0' or text is None:
                    method_description.append('None')
                else:
                    method_description.append(text)
                last_is_dd = True


# remove `""` in text
for i in range(len(method_description)):
    index_name[i] = re.sub('\"', '', index_name[i])
    index_description[i] = re.sub('\"', '', index_description[i])
    method_description[i] = re.sub('\"', '', method_description[i])
    index_name[i] = re.sub('\n', '', index_name[i])
    index_description[i] = re.sub('\n', '', index_description[i])
    method_description[i] = re.sub('\n', '', method_description[i])


df = pd.DataFrame({'index_name': index_name, 'index_description': index_description,
                  'method_description': method_description})
df.to_csv('../../data/java-api/java_api.csv',
          index=False, encoding='utf-8')

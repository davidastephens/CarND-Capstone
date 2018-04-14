import bs4
import requests

sess = requests.session()
resp = sess.get('https://hci.iwr.uni-heidelberg.de/node/6132/download/5a09e7a88ac00846a95d3027c817f824')
soup = bs4.BeautifulSoup(resp.content)
            
        
           
def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = sess.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename

for link in soup.find_all('a'):
    if link.has_attr('type'):
       link = link.attrs['href']
       if 'zip' in link and 'rgb' in link:
           download_file(link)
           
           
           


from html.parser import HTMLParser
import urllib
import urllib.request
import requests
from urllib.error import HTTPError

class WikiParser(HTMLParser):
	def __init__(self):
		super(WikiParser, self).__init__()

		self.tab_count = 0

		self.links = []

	def handle_starttag(self, tag, attrs):

		if tag == 'a':
			info = dict(attrs)
			# if 'class' not in info or info['class'] !=
			self.links.append(info)
			#self.printer('***T: {} {}'.format(tag, str(info)))
		else:
			#self.printer('T: {}'.format(tag))
			pass
		#self.tab_count += 1

	# def handle_endtag(self, tag):
	# 	pass
	# 	#self.printer('E: {}'.format(tag))
	# 	#self.tab_count -= 1
	#
	# def handle_data(self, data):
	# 	pass
	# 	#self.printer(data)

	def printer(self, s):
		# print('{}{}'.format('  '*self.tab_count,s))
		pass


bad_headers = {'File', 'Book', 'Wikipedia', 'Wikipedia_talk', 'User', 'User_talk', 'Template', 'Template_talk', 'Help', 'Talk', 'Category', 'Special', 'Portal'}

def link_cond(link):
	if 'class' in link:
		return False
	if 'href' not in link:
		return False
	if 'accesskey' in link:
		return False
	if link['href'][:6] != '/wiki/':
		return False

	if ':' in link['href']:
		idx = link['href'].find(':')
		header = link['href'][6:idx]

		if header in bad_headers:
			return False
		#else:
			#print('Bad header might be missing: {}'.format(link['href'][6:]))
			#print('BH: {}'.format(link['href'][6:]))

	return True

def category_cond(link):
	if 'href' not in link:
		return False
	if link['href'][:15] != '/wiki/Category:':
		return False
	return True

def redirect_cond(link):
	if 'class' not in link:
		return False
	if 'accesskey' in link:
		return False
	if link['class'] != 'mw-redirect':
		return False
	if ':' in link['href']:
		idx = link['href'].find(':')
		header = link['href'][:idx]
		if header in bad_headers:
			return False
	return True

def download_article_html_urllib(article_name):
	url = "https://en.wikipedia.org/wiki/{}".format(
		article_name.decode())
	
	#url = "https://en.wikipedia.org/wiki/{}".format(
	#	article_name.decode() if isinstance(article_name, bytes) else article_name)
	
	content = urllib.request.urlopen(url).read()
	return content


def download_article_html_requests(article_name):
	url = "https://en.wikipedia.org/wiki/{}".format(
		article_name.decode())

	# url = "https://en.wikipedia.org/wiki/{}".format(
	#	article_name.decode() if isinstance(article_name, bytes) else article_name)

	resp = requests.get(url,)# headers = {'User-agent': 'your bot 0.1'})

	if resp.status_code != 200:
		raise HTTPError(url, resp.status_code, 'Error* {}: unknown error'.format(resp.status_code), None, None)

	content = resp.text
	return content

def wiki_expand(article_name, download_fn=None): # collects only text links and redirects

	if download_fn is None:
		download_fn = download_article_html_requests

	content = download_fn(article_name)

	parser = WikiParser()
	parser.feed(str(content))
	links = [link for link in parser.links if link_cond(link)]
	redirects = [redir for redir in parser.links if redirect_cond(redir)]
	cats = [cat for cat in parser.links if category_cond(cat)]
	# len(links), len(redirects), len(cats)

	return links, redirects, cats, len(content)

QUERY_URL = "https://en.wikipedia.org/w/api.php"
def wiki_query(param, get_all=True):
	
	prev = {}
	out = []
	
	while True:
		
		param.update(prev)
	
		R = requests.get(url=QUERY_URL, params=param)

		try:
			DATA = R.json()
		except Exception as e:
			raise HTTPError(R.url, R.status_code, 'Error* {}: unknown error'.format(R.status_code), None, None)

		if not get_all:
			return DATA
		
		out.append(DATA['query'])
		
		if 'continue' in DATA:
			prev = DATA['continue']
		if 'batchcomplete' in DATA:
			break
			
	return out

def get_redirs(name):
	PARAMS = {
		'action': "query",
		'format': "json",
		'rdnamespace': '0',
		'titles': name,
		'prop': "redirects",
		'rdlimit': 'max',
	}
	raws = wiki_query(PARAMS)
	
	entries = [next(iter(raw['pages'].values())) for raw in raws]
	
	ID = entries[0]['pageid'] if 'pageid' in entries[0] else -1
	title = entries[0]['title']
	
	redirs = []
	for e in entries:
		if 'redirects' not in e:
			return ID, title, None
		redirs.extend([rec for rec in e['redirects'] if rec['ns'] == 0])
		
	return ID, title, redirs
	
def get_all_redirs(*names):
	assert len(names) <= 50, '{} too long'.format(len(names))
	
	titles = b'|'.join(names)
	
	PARAMS = {
		'action': "query",
		'format': "json",
		'titles': titles,
		'redirects': True,
	}
	
	R = requests.get(url=QUERY_URL, params=PARAMS)
	
	try:
		DATA = R.json()
	except Exception as e:
		raise HTTPError(R.url, R.status_code, 'Error* {}: unknown error'.format(R.status_code), None, None)
	
	if 'query' not in DATA or 'redirects' not in DATA['query']:
		return []
	return DATA['query']['redirects']
	
	
def get_links(name):
	PARAMS = {  # get links
		"action": "query",
		"format": "json",
		'titles': name,
		"prop": "links",
		'redirects': True,
		'pllimit': 'max',
		'plnamespace': '0',
	}
	
	raws = wiki_query(PARAMS)
	
	entries = [next(iter(raw['pages'].values())) for raw in raws]
	
	links = []
	try:
	
		for e in entries:
			links.extend([rec for rec in e['links'] if rec['ns'] == 0])
	except:
		pass
	return links

def wiki_collect(name): # name of either redirect or article
	
	ID, title, redirs = get_redirs(name)
	
	if redirs is None:
		return ID, title, None, None
	
	links = get_links(name)
	
	return ID, title, links, redirs
	


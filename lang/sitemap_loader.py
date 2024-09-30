### it is a scraper which is more advancer than the webbase loader as it can scrape the whole data from a website from all the pages of the website. just you have to put sitemap_index.xml at the end of the website url and then pass it to the sitemap loader. Actually the sitemap is the list of all the urls of all the pages of the website. And also the webbase loader will give the data in a single document class while sitemap loader will have multiple docs. Here is the code to do this:

import nest_asyncio

nest_asyncio.apply()

from langchain_community.document_loaders.sitemap import SitemapLoader
sitemap_loader = SitemapLoader(web_path="https://api.python.langchain.com/sitemap.xml")
docs = sitemap_loader.load()
#to find the length of docs
len(docs)
#to see individual doc
docs[0]
#to see all docs been scraped
docs
#_________________________________________________________________________________________________________________________
#to lazily load the docs use this:

lazy_loader = sitemap_loader.lazy_load()

first_item = next(lazy_loader)
first_item

second_item = next(lazy_loader)
second_item
#________________________________________________________________________________________________________________________
#we can also filter the urls:
loader = SitemapLoader(
    web_path="https://api.python.langchain.com/sitemap.xml",
    filter_urls=["https://api.python.langchain.com/en/latest"],
)
documents = loader.load()
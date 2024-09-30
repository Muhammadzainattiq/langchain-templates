#This loader can scrape the data from a webiste by recursively root url and all the children urls in it and parse them into documents.It can scrape more content than sitemap loader bcz The RecursiveUrlLoader recursively follows links from a starting URL, scraping all linked pages. This allows it to discover and load content that may not be included in the sitemap.
#The RecursiveUrlLoader returns the raw HTML content of each page, while the SitemapLoader parses the HTML and returns a structured Document object for each page.

from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    "https://docs.python.org/3.9/",
    # max_depth=2,
    # use_async=False,
    # extractor=None,
    # metadata_extractor=None,
    # exclude_dirs=(),
    # timeout=10,
    # check_response_status=True,
    # continue_on_failure=True,
    # prevent_outside=True,
    # base_url=None,
    # ...
)

docs = loader.load()
docs
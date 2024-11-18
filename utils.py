import requests
from bs4 import BeautifulSoup

def web_search_and_process(entity, query):
    """Perform web search and extract information."""
   
    api_url = f"https://www.bing.com/search?q={query}"
    response = requests.get(api_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if "http" in a["href"]]
        return links[:5] 
    else:
        return []

def parse_results(links, llm_pipeline):
    """Use LLM to process web search results."""
    extracted_data = []

    for link in links:
        response = requests.get(link)
        if response.status_code == 200:
            text = BeautifulSoup(response.text, "html.parser").get_text()
            result = llm_pipeline(f"Extract key information from this text: {text[:500]}")
            extracted_data.append(result[0]["generated_text"])

    return {"Top Information": " ".join(extracted_data[:3])}

from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from openai import AzureOpenAI
from dotenv import load_dotenv
from functools import lru_cache
from pinecone import Pinecone
import os
from langchain_openai import AzureOpenAIEmbeddings
import streamlit as st

load_dotenv()

app = Flask(__name__)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

embeddings = AzureOpenAIEmbeddings(
    model= os.getenv("EMBEDDING_MODEL"),
    api_key = os.getenv("EMB_AZURE_OPENAI_API_KEY"),  
    api_version = os.getenv("OPENAI_API_VERSION"),
    azure_endpoint =os.getenv("EMB_AZURE_OPENAI_ENDPOINT") 
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("INDEX_NAME")
index = pc.Index(index_name)

combined_sys_inst = """**Objective:**  
Generate precise, context-rich questions for vector search based on Coca-Cola’s reports and industry trends, and extract meaningful, concise, and actionable insights from news articles. Focus on financial performance, market dynamics, competitive strategies, consumer behavior, and sustainability.

**Guidelines:**  
- **Relevance:** Questions and insights must directly relate to Coca-Cola’s operations, financials, or industry context.  
- **Contextual Depth:** Questions and insights should reflect an understanding of relationships between metrics, strategies, and trends.  
- **Variety:** Include factual, inferential, summarization, and contextual questions.  
- **Clarity:** Questions and insights should be concise and unambiguous.  
- **Searchability:** Phrase questions to effectively retrieve relevant information in a vector search system.  
- **Summarization:** Provide clear and concise summaries of key points in articles.  
- **Insight Generation:** Identify trends, implications, and underlying themes.  
- **Categorization:** Classify content based on topics (e.g., Politics, Economy, Technology, Environment).  
- **Fact Emphasis:** Highlight critical statistics, data points, or factual claims.  
- **Sentiment Analysis:** Determine the tone or bias of the article (e.g., optimistic, pessimistic, neutral).  

**Process:**  
1. **Understand the Articles:** Analyze news articles, identifying key metrics, initiatives, risks, and trends.  
2. **Identify Key Themes:** Focus on financial performance, market trends, competitive strategies, consumer preferences, and sustainability.  
3. **Generate Questions:**  
   - **Factual:** "What was Coca-Cola’s net revenue growth in Q3 2023?"  
   - **Inferential:** "How might Coca-Cola’s expansion into health-focused beverages affect its long-term profitability?"  
   - **Summarization:** "Summarize Coca-Cola’s 2023 sustainability goals and progress."  
   - **Contextual:** "How does Coca-Cola’s pricing strategy align with current inflationary trends?"  
4. **Extract Details:**  
   - **Summarization:** Provide a 3-5 sentence summary of the article.  
   - **Insights:** List 2-3 bullet points detailing implications or actionable takeaways.  
   - **Category:** Clearly state the topic(s) or industry affected.  
   - **Sentiment:** Indicate the sentiment and justify it briefly.  
5. **Review and Refine:** Ensure questions and insights are clear, relevant, and optimized for vector search and actionable insights.

**Output Format:**  
- **Questions:** Provide 4-5 context-rich questions, each on a new line.  
- **Details:**  
  - **Headline:** [Insert headline here]  
  - **Summary:** [3-5 concise sentences summarizing the article.]  
  - **Insights:**  
    - Bullet point 1: [Key insight or implication.]  
    - Bullet point 2: [Additional insight or takeaway.]  
  - **Category:** [Insert relevant category, e.g., "Technology" or "Global Economy."]  
  - **Sentiment:** [Insert sentiment, e.g., "Neutral with an optimistic outlook."]  
"""

# Cache for visited links to avoid redundant requests
@lru_cache(maxsize=100)
def visit_link(link):
    try:
        response = requests.get(link, timeout=10)
        
        # Handle common HTTP errors
        error_codes = {
            403: "403 Forbidden",
            404: "404 Not Found",
            500: "500 Internal Server Error",
            401: "401 Unauthorized"
        }
        
        if response.status_code in error_codes:
            return error_codes[response.status_code], []
        
        if response.status_code >= 400:
            return f"{response.status_code} Client/Server Error", []

        if 'text/html' not in response.headers.get('Content-Type', ''):
            return "Non-HTML Content", []

        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No title found"
        paragraphs = [p.get_text() for p in soup.find_all('p')]

        if not title or not paragraphs:
            return "Empty or Invalid Content", []

        return title, paragraphs

    except requests.RequestException as e:
        return f"Request Exception: {str(e)}", []

def search_google(api_key, cx, query):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cx,
        'q': query,
        'num': 2
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if 'items' not in data:
            return [], [], []

        urls = [item['link'] for item in data['items']]
        all_titles = []
        all_paragraphs = []
        valid_urls = []

        for url in urls:
            title, paragraphs = visit_link(url)
            
            if not isinstance(title, str) or title.startswith(("403", "404", "500", "401", "Request Exception")):
                continue

            all_titles.append(title)
            all_paragraphs.append(paragraphs)
            valid_urls.append(url)

        return all_titles, all_paragraphs, valid_urls

    except requests.RequestException as e:
        return [], [], []
    
def retriever_generator(query, summary):
    try:
        query_embedding = embeddings.embed_query(query)
        result = index.query(top_k=2, vector=query_embedding, include_metadata=True, namespace="practus-cococola-demo")
        context = "".join([res.metadata['content'] for res in result.matches])
    except Exception as e:
        return f"Error retrieving from pinecone: {e}"
    
    try:
        response = client.chat.completions.create(
            model="Alfred-gpt-4-o-mini",
            messages=[
                {"role": "system", "content": "You are Alfred specially designed for coco cola company. You need to generate 4-5 line answer and insights on what we can do for the question from the given context and news summary."},
                {"role": "user", "content": f"User query: {query} \n news summary: {summary} \n context: {context}"}
            ]
        )
        return f"`{response.choices[0].message.content}"
    except Exception as e:
        return f"Error generating insights: {e}"

def generate_insights(content):
    try:
        response = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            messages=[
                {"role": "system", "content": combined_sys_inst},
                {"role": "user", "content": content}
            ]
        )
        x = response.choices[0].message.content.split('**Details:**')
        y=x[0].split('**Questions:**')[1].split('\n')
        cleaned_questions = [item.strip() for item in y if item.strip()]
        
        # return (x[1], cleaned_questions)
    
    except Exception as e:
        return f"Error generating insights: {str(e)}"
    
    try:
        ans=[]
        if cleaned_questions:
            for query in cleaned_questions:
                ans.append(retriever_generator(query, x[1]))
                
        return (x[1], ans)
    except Exception as e:
        return f"Error generating insights: {e}"
    
    
def process_generate_insights_parallely(contents):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_insights, content) for content in contents]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def fetch_news_insights(query):
    api_url = os.getenv("FLASK_API_URL", "http://localhost:5000/news-insights")
    response = requests.post(api_url, json={"query": query})

    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        return {"error": response.json().get("error", "Unknown error")}


@app.route('/news-insights', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400

        api_key = os.getenv('GOOGLE_API_KEY')
        cx = os.getenv('GOOGLE_CX')
        
        if not api_key or not cx:
            return jsonify({'error': 'Missing Google API credentials'}), 500

        titles, paragraphs, urls = search_google(api_key, cx, data['query'])
        
        if not titles:
            return jsonify({'error': 'No results found'}), 404

        llm_content = [
            f'Title: {title}\ncontent: {"".join(content)}'
            for title, content in zip(titles, paragraphs)
        ]
        
        question_insights_pair = process_generate_insights_parallely(llm_content)
            
        
        return jsonify({
            'results': [
                {
                    'title': title,
                    'url': url,
                    'summary': summary[0],
                    'insights_from_reports': summary[1]
                }
                for title, url, summary in zip(titles[:5], urls[:5], question_insights_pair)
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    st.title("News Insights Explorer")
    st.write("Enter a query to fetch insights and summaries from recent news articles.")
    results = []
    query = st.text_input("Enter your query:")

    if st.button("Get Insights") and query:
        st.write("Fetching insights... 🔍")
        results = fetch_news_insights(query)

        if "error" in results:
            st.error(results["error"])
        else:
            for item in results:
                with st.expander(item["title"]):
                    st.subheader("Insights from internal Reports")
                    st.write(item["insights_from_reports"])
                    st.subheader("Summary from News")
                    st.write(item["summary"])
                    st.markdown(f"[Read more]({item['url']})", unsafe_allow_html=True)
        st.write("Above are the fetched insights...")
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)), use_reloader=False)

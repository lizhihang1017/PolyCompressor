# prompt.py


Evidence_Extractor_Prompt = """
I will provide you with {num} sentences, each indicated by a numerical identifier []. Select the sentences based on their relevance to the search query: {question}

{context}

Search Query: {question}  

Please follow the steps below: 
Step 1. Please list up the information requirements to answer the query. 
Step 2. for each requirement in Step 1, find the sentences that has the information of the requirement. 
Step 3. Choose the sentences that mostly covers clear and diverse information to answer the query. Number of sentences is unlimited. 

The format of final output should be ‘### Final Selection: [] []’, e.g., ### Final Selection: [2] [1].
"""

Probe_Queries_Prompt = """
Please perform the following task for the question: {question}

Generate three synonymous questions from the above, requiring the application of synonym replacement, syntactic variation, and semantic expansion methods, respectively.

You must output exactly three questions in the following format, with no other text:
[1] First question here
[2] Second question here
[3] Third question here
"""

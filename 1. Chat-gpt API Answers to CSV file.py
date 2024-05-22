
import os
os.environ["OPENAI_API_KEY"]=private api key
import openai
import pandas as pd
file = open('Decision questions.csv')
df = pd.read_csv(r'C:/Users/shing/OneDrive - email.ucr.edu/Desktop/API Chat-GPT/Decision questions.csv')
#from langchain.agents.agent_types import AgentType
#from langchain_experimental.agents.agent_toolkits import create_csv_agent
#from langchain_openai import ChatOpenAI, OpenAI
#from langchain.prompts import PromptTemplate
#from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser


gpt_questions = [] #list of questions for GPT
gpt_short_answers = [] #list of answers for GPT
gpt_raw_answers = [] #list of raw answers from GPT
binary_answer = [] #list of binary answers from GPT


for n, row in df.iterrows(): #loop through rows of dataframe
  #extract question information from relevant columns in each row
  ss_amount = row['Smaller sooner amount ($)']
  ss_delay = row['Smaller sooner delay (days)']
  ll_amount = row['Larger later amount ($)']
  ll_delay = row['Larger later delay (days)']
  question_str = 'Would you rather have $' + str(int(ss_amount)) + ' in ' + str(int(ss_delay)) \
    + ' days or $' + str(int(ll_amount)) + ' in ' + str(int(ll_delay)) + ' days? Answer without explanation.'
  question_str = question_str.replace('in 0 days','now') #1. question has been created.

  gpt_questions.append(question_str)                     #2. question has been added to a list

  response = openai.chat.completions.create(
      model="gpt-4-turbo",
      messages = [
          {"role": "system", "content": "Pick one."},
          {"role": "user", "content": question_str},          #3. Chatgpt runs through the question just created saved in gpt_questions
          ])
  gpt_raw_answers.append(response.choices[0].message.content)       #4. Gpt's answer has been added to the gpt_raw_answers list
  if str(int(ll_amount)) in response.choices[0].message.content:
    response.choices[0].message.content = "LL"
    binary_answer.append("1")
  elif str(int(ss_amount)) in response.choices[0].message.content:
    response.choices[0].message.content = "SS"
    binary_answer.append("0")
  gpt_short_answers.append(response.choices[0].message.content)

df['gpt_questions'] = gpt_questions
df['gpt_raw_answers'] = gpt_raw_answers
df['SS or LL'] = gpt_short_answers
df['binary_answer'] = binary_answer
df.to_csv('Decision questions.csv')



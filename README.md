# SynoLangchain V1.1

using synology chat with LLMs and Langchain

Only tested on Windows 10, builds on llama-ccp-python 


Install
  
  1) install visual studio community 2022 (I checked python development and C++ developement)
  
  2) clone repository
  
  3) create virtual envirement in folder    
    
    python -m venv venv
  
  4) activate virual envirement             
  
    venv/Scripts/activate
 
  5) install the requirements
    
    pip install -r requirements.txt
     
Setup

  1) place your LLM in the model folder and copy that file name to the settings file
  
  2) setup a new bot in your synology chat app
  
  3) copy the Token and the incoming URL to the settings file
  
  4) the outgoing URL in synology integration will be http://IP_ADDRESS:FLASK_PORT/SynologyLLM change IP_ADDRESS and FLASK_PORT to what it is on your local PC your running the model on
  
  5) Use either SynoLangchain.bat file or command
  
    python SynoLangchain.py

Features
  
  1) Loads any llama.cpp model that is supported
  2) uses langchain to integrate chromadb to provide a vectorstore database
  3) uses langchain to load a pdf or txt file and searches it (RAG)
     use folder /memorydb/knowldege to put your files
  5) uses langchain to search wikipedia
  6) Can add or delete core memories which are just input/output conversations in /memorydb/core_memories folder

Commands

  1) /commands will list commands
  2) /mode there is chat, rag, wiki, write
  3) write mode has a /continue funtion to reinput what it just output to finish the writting

Modes

  chat mode
  
     1) is the normal chat prompt mode but it has a vector store for long term memory and a entity store for individual info

  rag mode
  
    1) this is the talk with your documents mode will load up either txt files or pdf's

  wiki mode
  
    1) this is the ask wikipedia anything mode and it will search wikipedia for an answer

  write mode
  
    1) this the input goes directly into the llm no prompt mode used for completeions and I find it works best for creativity stores and such

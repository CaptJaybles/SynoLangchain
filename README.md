# SynoLangchain

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
  
Optional GPU Support for CUBLAS 

  1) Open powershell in folder that has all the files
    
    python -m venv venv

    pip uninstall -y llama-cpp-python

    $Env:LLAMA_CUBLAS = "1"
     
    $Env:FORCE_CMAKE = "1"
     
    $Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
     
    pip install llama-cpp-python --no-cache-dir

  2) change GPU layers in settings to what will fit your GPU
     
Setup

  1) place your LLM in the model folder and copy that file name to the settings file
  
  2) setup a new bot in your synology chat app
  
  3) copy the Token and the incoming URL to the settings file
  
  4) the outgoing URL in synology integration will be http://IP_ADDRESS:FLASK_PORT/synologyLLM change IP_ADDRESS and FLASK_PORT to what it is on your local PC your running the model on
  
  5) Use either synolangchain.bat file or command
  
    python synolangchain.py

Features
  
  1) Loads any llama.cpp model that is supported
  2) uses langchain to provide a vector memory
  3) uses langchain to load a pdf or txt file and searches it
     use folder /memorydb/knowldege to put your files
  5) uses langchain to search either wikipedia or duckduckgo
  6) Can add or delete core memories which are just input/output conversations in /memorydb/core_memories folder

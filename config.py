class Config:
    # Model settings
    PATIENT_MODEL = "gpt-4o-mini-2024-07-18"  # Patient uses GPT-4o mini
    # 1
    #PATIENT_API_KEY = "YOUR_OpenAI_API_KEY"  # ChatGPT API Key
        
   ################ Gemini doctor model selection, available options: gemini-2.0-flash-exp/gemini-1.5-flash-002/gemini-1.5-pro  ################
   
    #DOCTOR_GEMINI_MODEL = "gemini-1.5-pro" 
    #DOCTOR_GEMINI_MODEL = "gemini-1.5-flash-002"  
    DOCTOR_GEMINI_MODEL = "gemini-1.5-flash-8b"
    
    #DOCTOR_GEMINI_API_KEY = 'YOUR_Gemini_Key' #1
    

    ### ChatGPT Doctor model and API Key selection ################################
    #DOCTOR_CHATGPT_MODEL = "gpt-4o-2024-11-20" 
    #DOCTOR_CHATGPT_MODEL = "gpt-4o-mini-2024-07-18" 
    DOCTOR_CHATGPT_MODEL = "gpt-3.5-turbo-0125" 
    
    DOCTOR_CHATGPT_API_KEY = 'YOUR_OpenAI_KEY2'  
            
    # Test case file path
    CASE_FILE_PATH = r'F:\Python\USMLE_restructured.jsonl'
    
    # Timing settings (seconds)
    RESPONSE_DELAY = 6  # Response generation wait time
    CASE_INTERVAL = 7   # Interval between cases
    
    # Dialogue settings
    DIALOGUE_ROUNDS = 3  # Number of dialogue rounds per case
    
           
    # Scoring settings
    MIN_SCORE = 1
    MAX_SCORE = 10
   
    # Debug mode configuration
    DEBUG = False  # Debug mode off by default

  
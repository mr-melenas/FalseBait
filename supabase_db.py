import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def guardar_fila_completa(data_dict):
    try:
        response = supabase.table("phishing_data").insert(data_dict).execute()
        print("Fila guardada correctamente en Supabase.")
        return response
    except Exception as e:
        print("Error al guardar en Supabase:", e)
        return None

def convertir_a_dict(url, domain, prediccion, *features):
    
    data_dict = {
        "created_at": datetime.now(), 
        "FILENAME": features[0], 
        "URL": url,
        "URLLength": len(url),
        "Domain": domain,
        "DomainLength": len(domain),
        "IsDomainIP": features[1], 
        "TLD": features[2], 
        "URLSimilarityIndex": 100,  
        "CharContinuationRate": features[3],  
        "TLDLegitimateProb": features[4],  
        "URLCharProb": features[5],  
        "TLDLength": len(features[2]), 
        "NoOfSubDomain": domain.count('.') - 1,  
        "HasObfuscation": 1 if features[6] else 0, 
        "NoOfObfuscatedChar": len(features[7]),  
        "ObfuscationRatio": round(len(features[7]) / len(url), 3) if len(url) > 0 else 0.0,
        "NoOfLettersInURL": sum(c.isalpha() for c in url),
        "LetterRatioInURL": round(features[8], 3), 
        "NoOfDegitsInURL": sum(c.isdigit() for c in url),
        "DegitRatioInURL": round(features[9], 3),  
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": len(features[10]),  
        "SpacialCharRatioInURL": round(features[11], 3), 
        "IsHTTPS": 1 if url.startswith("https://") else 0,
        "LineOfCode": features[12], 
        "LargestLineLength": features[13], 
        "HasTitle": 1 if features[14] else 0, 
        "Title": features[14],
        "DomainTitleMatchScore": 1 if features[15] and domain.lower() in features[14].lower() else 0,
        "URLTitleMatchScore": features[16],  
        "HasFavicon": features[17],
        "Robots": features[18], 
        "IsResponsive": 0, 
        "NoOfURLRedirect": features[19], 
        "NoOfSelfRedirect": features[20], 
        "HasDescription": features[21], 
        "NoOfPopup": features[22],
        "NoOfiFrame": features[23], 
        "HasExternalFormSubmit": features[24], 
        "HasSocialNet": features[25], 
        "HasSubmitButton": features[26], 
        "HasHiddenFields": features[27], 
        "HasPasswordField": features[28], 
        "Bank": features[29], 
        "Pay": features[30],  
        "Crypto": features[31], 
        "HasCopyrightInfo": 1 if "©" in features[32] else 0, 
        "NoOfImage": features[33],  
        "NoOfCSS": features[34], 
        "NoOfJS": features[35],  
        "NoOfSelfRef": features[36],  
        "NoOfEmptyRef": features[37],  
        "NoOfExternalRef": features[38], 
        "label": prediccion
    }
    
    return data_dict

def guardar_datos(url, domain, prediccion, *features):

    data_dict = convertir_a_dict(url, domain, prediccion, *features)
    return guardar_fila_completa(data_dict)

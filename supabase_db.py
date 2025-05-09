import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
# Initialize Supabase client

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
# Save data to Supabase
def save_completa(data_dict):
    try:
        response = supabase.table("phishing_inputs").insert(data_dict).execute()
        print("Fila guardada correctamente en Supabase.")
        return response
    except Exception as e:
        print("Error al guardar en Supabase:", e)
        return None

def save_fill_complete(features, prediction, url, filename, title, domain):
    try:
        data_dict = {
            "created_at": datetime.now().isoformat(),
            "FILENAME": filename,
            "URL": url,
            "Domain": domain,
            "label": prediction,
            **{key: features[key] for key in [
            "URLLength", "IsDomainIP", "TLD", "URLSimilarityIndex", "CharContinuationRate",
            "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation",
            "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL",
            "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
            "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
            "IsHTTPS", "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
            "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive", "NoOfURLRedirect",
            "NoOfSelfRedirect", "HasDescription", "NoOfPopup", "NoOfiFrame", "HasExternalFormSubmit",
            "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField", "Bank",
            "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
            "NoOfEmptyRef", "NoOfExternalRef"
            ]},
            "Title": title,
            "DomainLength": features["URLLength"]
        }
        #print("data_dict:", data_dict)
        save_completa(data_dict)
        #return response
    except Exception as e:
        print("BD Error while saving data:", e)

        return None

#def save_fill_complete_B(url, domain, prediccion, *features):

    #data_dict = convertir_a_dict(url, domain, prediccion, *features)
    #return guardar_fila_completa(data_dict)
